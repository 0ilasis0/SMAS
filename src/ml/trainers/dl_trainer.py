import copy
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from base import MathTool, MLTool
from debug import dbg
from ml.const import DLModelType, DLParamKey, ModelAttr
from ml.params import DLHyperParams, TrainConfig
from ml.trainers.dl_net import DLModelFactory, RNNType


class DLTrainer:
    ''' DL 離線訓練器 (與 XGBTrainer API 對齊) '''
    def __init__(self, ticker: str, dl_model_type: DLModelType, rnn_type: RNNType, custom_hp: dict = None):
        self.dl_model_type = dl_model_type
        self.rnn_type = rnn_type
        self.ticker = ticker
        self.device = self._detect_device()

        dbg.log(f"DLTrainer 初始化 [{self.ticker} - {self.dl_model_type}]")

        if custom_hp:
            self.batch_size = custom_hp.get(DLParamKey.BATCH_SIZE, DLHyperParams.batch_size)
            self.epochs = custom_hp.get(DLParamKey.EPOCHS, DLHyperParams.epochs)
            self.learning_rate = custom_hp.get(DLParamKey.LEARNING_RATE, DLHyperParams.learning_rate)
        else:
            self.batch_size = DLHyperParams.batch_size
            self.epochs = DLHyperParams.epochs
            self.learning_rate = DLHyperParams.learning_rate

        self.optimal_epochs = self.epochs

        # 平均 AUC，預設為0.5
        self.cv_avg_auc: float = 0.5

    def _detect_device(self):
        """自動偵測是否支援 GPU 加速 or Mac 晶片加速"""
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def train_with_cv(self, X_raw: np.ndarray, y: np.ndarray, original_index: pd.Index, lookahead: int, n_splits: int = TrainConfig.N_SPLITS) -> pd.Series:
        n_splits = MathTool.clamp(n_splits, TrainConfig.N_SPLITS_MIN, TrainConfig.N_SPLITS_MAX)
        dbg.log(f"開始執行 DL (CNN-{self.rnn_type.value if self.rnn_type else '1DCNN'}) 嚴格三階段交叉驗證 (Fold={n_splits}, Gap={lookahead})...")

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=lookahead)
        oof_predictions = pd.Series(index=original_index, dtype=float)

        # 帶入 DataManager 裡定義的指標總數
        num_features = X_raw.shape[2]
        cv_accuracies, cv_aucs = [], []
        best_epochs = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
            X_train_raw, y_train = X_raw[train_idx], y[train_idx]
            X_val_raw, y_val = X_raw[val_idx], y[val_idx]

            scaler = RobustScaler()

            # 將 3D 壓平讓 Scaler 學習，轉換並膨脹回原來的 3D 形狀
            X_train = scaler.fit_transform(X_train_raw.reshape(-1, num_features)).reshape(X_train_raw.shape)
            X_val = scaler.transform(X_val_raw.reshape(-1, num_features)).reshape(X_val_raw.shape)

            train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

            model = DLModelFactory.create(
                model_type=self.dl_model_type,
                num_features=num_features,
                time_steps=DLHyperParams.time_steps,
                rnn_type=self.rnn_type
            ).to(self.device)

            # 負責衡量模型的預測與實際答案之間的距離
            pos_weight_val = MLTool.calculate_scale_weight(y_train)
            pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

            # 優化器
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            # 當學習不下去時，自動調整學習率
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=DLHyperParams.scheduler_factor, patience=DLHyperParams.scheduler_patience
            )

            best_val_loss = float('inf')
            patience_counter = 0
            best_epoch_for_fold = 0
            best_model_wts = copy.deepcopy(model.state_dict())

            # --- Training Loop ---
            for epoch in range(self.epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(X_batch).view(-1, 1), y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # --- Validation (Early Stopping) ---
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    # 使用放大的 val_loader，避免微氣候干擾
                    for X_v, y_v in val_loader:
                        X_v, y_v = X_v.to(self.device), y_v.to(self.device)
                        val_loss += criterion(model(X_v).view(-1, 1), y_v).item()

                avg_val_loss = val_loss / len(val_loader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_epoch_for_fold = epoch + 1
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= TrainConfig.DL_EARLY_STOP_ROUND:
                        break

                scheduler.step(avg_val_loss)

            best_epochs.append(best_epoch_for_fold)

            # --- 收集 OOF 預測 ---
            model.load_state_dict(best_model_wts)
            model.eval()
            test_preds = []
            with torch.no_grad():
                for X_t, _ in val_loader:  # 直接對 val_loader 進行推論
                    preds = torch.sigmoid(model(X_t.to(self.device)).view(-1)).cpu().numpy()
                    preds_unscaled = MLTool.unscale_probability(preds, pos_weight_val)
                    test_preds.extend(np.atleast_1d(preds_unscaled))

            test_preds = np.array(test_preds)
            oof_predictions.iloc[val_idx] = test_preds

            # --- 計算指標 ---
            y_pred_binary = (test_preds > 0.5).astype(int)
            acc = accuracy_score(y_val, y_pred_binary)
            cv_accuracies.append(acc)

            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, test_preds)
                cv_aucs.append(auc)
                auc_str = f"{auc:.4f}"
            else:
                auc_str = "N/A"

            dbg.log(f"Fold {fold+1}: Accuracy={acc:.4f}, AUC={auc_str} (最佳 Epoch: {best_epoch_for_fold})")

        if best_epochs:
            self.optimal_epochs = int(np.mean(best_epochs))
            dbg.log(f"💡 CV 判定最佳平均 Epoch 數為: {self.optimal_epochs} (原設定 {self.epochs})")

        if not cv_accuracies:
            dbg.error("交叉驗證失敗：資料量不足。")
            return pd.Series(dtype=float)

        avg_auc = np.mean(cv_aucs) if cv_aucs else 0.5
        self.cv_avg_auc = avg_auc
        dbg.log(f"【DL 驗證結果】平均 Accuracy: {np.mean(cv_accuracies):.4f}, 平均 AUC: {avg_auc:.4f}")

        # if len(np.unique(y_val)) > 1:
        #     auc = roc_auc_score(y_val, test_preds)
        #     cv_aucs.append(auc)
        #     auc_str = f"{auc:.4f}"

        #     # 呼叫排列重要性計算器
        #     fold_importances = self._calculate_permutation_importance(model, X_val, y_val, base_auc=auc)

        #     dl_feature_names = FeatureCol.get_dl_features()

        #     # 安全防呆：確保陣列長度與名稱數量一致
        #     if len(dl_feature_names) == len(fold_importances):
        #         imp_series = pd.Series(fold_importances, index=dl_feature_names).sort_values(ascending=False)

        #         dbg.log(f"\n🧠 【DL 模型 (Fold {fold+1}) 核心特徵 (Top 5)】")
        #         for idx, (feat_name, imp_score) in enumerate(imp_series.head(5).items(), 1):
        #             dbg.log(f"  {idx}. {feat_name}: AUC 貢獻 {imp_score:.4f}")

        #         dbg.log(f"\n🗑️ 【DL 模型 (Fold {fold+1}) 最沒用特徵 (Bottom 5)】")
        #         for idx, (feat_name, imp_score) in enumerate(imp_series.tail(5).items(), 1):
        #             dbg.log(f"  倒數 {6-idx}. {feat_name}: AUC 貢獻 {imp_score:.4f}")
        #         dbg.log("-" * 40)
        #     else:
        #         dbg.war(f"特徵數量不匹配！(X_val 特徵數: {len(fold_importances)}, FeatureCol 數量: {len(dl_feature_names)})")
        # else:
        #     auc_str = "N/A"

        return oof_predictions.dropna()

    def _calculate_permutation_importance(self, model: nn.Module, X_val: np.ndarray, y_val: np.ndarray, base_auc: float) -> np.ndarray:
        """
        深度學習特徵重要性評估器
        藉由隨機打亂單一特徵，觀察 AUC 下降的幅度來逆向推導特徵重要性。
        """
        num_features = X_val.shape[2]
        importances = np.zeros(num_features)

        # 如果驗證集只有一種標籤，無法計算 AUC，直接回傳 0
        if len(np.unique(y_val)) <= 1:
            return importances

        model.eval()
        with torch.no_grad():
            for f_idx in range(num_features):
                # 複製一份乾淨的驗證集
                X_corrupted = X_val.copy()

                # 將這一個維度 (特徵) 的所有數據壓平、徹底洗牌、再塞回去
                # 這樣能完美摧毀該特徵的預測力，且不破壞其他特徵
                orig_shape = X_corrupted[:, :, f_idx].shape
                flat_feature = X_corrupted[:, :, f_idx].flatten()
                np.random.shuffle(flat_feature)
                X_corrupted[:, :, f_idx] = flat_feature.reshape(orig_shape)

                # 用這個「被破壞」的數據集重新推論
                corrupted_loader = self._create_dataloader(X_corrupted, y_val, shuffle=False)
                test_preds = []
                for X_t, _ in corrupted_loader:
                    preds = torch.sigmoid(model(X_t.to(self.device))).cpu().numpy()
                    test_preds.extend(np.atleast_1d(preds))

                # 計算破壞後的 AUC
                corrupted_auc = roc_auc_score(y_val, test_preds)

                # 重要性 = 基準 AUC - 破壞後 AUC (掉得越多，代表特徵越重要)
                importances[f_idx] = base_auc - corrupted_auc

        return importances

    def train_and_save_final_model(self, X_raw: np.ndarray, y: np.ndarray, valid_index: pd.Index, oof_preds: pd.Series, save_path: Path | str):
        dbg.log(f"開始訓練最終上線版 DL 模型 (動態 Epoch={self.optimal_epochs})...")
        # 形狀為 (Batch, Time_Steps, Features)
        num_features = X_raw.shape[2]
        final_scaler = RobustScaler()
        X_2d = X_raw.reshape(-1, num_features)
        X_scaled = final_scaler.fit_transform(X_2d).reshape(X_raw.shape)

        full_loader = self._create_dataloader(X_scaled, y, shuffle=True)

        pos_weight_val = MLTool.calculate_scale_weight(y)
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

        model = DLModelFactory.create(
            model_type=self.dl_model_type,
            num_features=num_features,
            time_steps=DLHyperParams.time_steps,
            rnn_type=self.rnn_type
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        model.train()
        for epoch in range(self.optimal_epochs):
            for X_batch, y_batch in full_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch).view(-1, 1), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        # 由於 y 是 numpy array，我們先將它轉為含有正確 index 的 Series 以利對齊
        y_series = pd.Series(y, index=valid_index)

        # 精準對齊 OOF 預測值與真實答案 (防範 dropna() 造成的長度差異)
        y_true_oof = y_series.loc[oof_preds.index].values
        y_prob_oof = oof_preds.values

        # 計算客觀無洩漏的真實 AUC
        val_auc = self.cv_avg_auc
        dbg.log(f"✅ DL 模型標定完成 | 真實 CV 平均 AUC: {val_auc:.4f}")

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            ModelAttr.STATE_DICT: model.state_dict(),
            ModelAttr.VAL_AUC: val_auc,
            ModelAttr.TRAIN_SCALE_WEIGHT: float(pos_weight_val)
        }
        torch.save(checkpoint, str(save_path_obj))

        dbg.log(f"最終模型權重 (含真實 OOF AUC 資訊) 已儲存至: {save_path_obj}")
        return final_scaler

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray | None = None, shuffle: bool = False) -> DataLoader:
        ''' y跟X釘在一起並轉成tensor後打包 (支援推論與 pin_memory 加速) '''
        X_tensor = torch.as_tensor(X, dtype=torch.float32)
        if y is not None:
            y_tensor = torch.as_tensor(y, dtype=torch.float32).view(-1, 1)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        use_pin_memory = (self.device.type == 'cuda')
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=use_pin_memory
        )

    def load_inference_model(self, num_features: int, model_path: Path | str) -> nn.Module:
        """ 載入訓練好的模型權重與 AUC (極限 Debug 追蹤版) """
        dbg.log(f"準備載入 DL 模型，路徑: {model_path}")
        dbg.log(f"傳入的特徵數量 num_features: {num_features}")

        try:
            model_path = Path(model_path)
            if not model_path.exists():
                dbg.error(f"❌ 深度學習模型載入失敗: 找不到檔案 {model_path}")
                return None

            dbg.log(f"開始建立模型架構 (Factory)...")
            model = DLModelFactory.create(
                model_type=self.dl_model_type,
                num_features=num_features,
                time_steps=DLHyperParams.time_steps,
                rnn_type=self.rnn_type
            ).to(self.device)

            checkpoint = torch.load(str(model_path), map_location=self.device, weights_only=False)
            dbg.log(f"硬碟檔案讀取成功！檔案類型: {type(checkpoint)}")

            model.load_state_dict(checkpoint.get(ModelAttr.STATE_DICT))
            model.val_auc = checkpoint.get(ModelAttr.VAL_AUC, checkpoint.get(ModelAttr.VAL_AUC))
            model.train_scale_weight = checkpoint.get(ModelAttr.TRAIN_SCALE_WEIGHT, checkpoint.get(ModelAttr.TRAIN_SCALE_WEIGHT))

            model.eval()
            dbg.log(f"✅ 成功載入 DL 模型 (紀錄 AUC: {model.val_auc:.4f}): {model_path}")
            return model

        except Exception as e:
            error_details = traceback.format_exc()
            dbg.error(f"🔥 DL 模型載入發生深層崩潰！\n{'-'*40}\n詳細錯誤追蹤:\n{error_details}\n{'-'*40}")
            raise e