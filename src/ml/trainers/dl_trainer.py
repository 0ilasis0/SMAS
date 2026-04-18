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
from ml.const import DLModelType, DLParamKey
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
            self.batch_size = custom_hp.get(DLParamKey.BATCH_SIZE, DLHyperParams.BATCH_SIZE)
            self.epochs = custom_hp.get(DLParamKey.EPOCHS, DLHyperParams.EPOCHS)
            self.learning_rate = custom_hp.get(DLParamKey.LEARNING_RATE, DLHyperParams.LEARNING_RATE)
        else:
            self.batch_size = DLHyperParams.BATCH_SIZE
            self.epochs = DLHyperParams.EPOCHS
            self.learning_rate = DLHyperParams.LEARNING_RATE

        self.optimal_epochs = self.epochs

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

        cv_accuracies, cv_aucs = [], []
        best_epochs = []
        num_features = X_raw.shape[2]

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
            split_point = len(val_idx) // 2
            early_stop_end = split_point - lookahead

            if early_stop_end <= 0 or split_point >= len(val_idx):
                dbg.war(f"Fold {fold+1}: 樣本數不足以切割三階段，跳過。")
                continue

            early_stop_idx = val_idx[:early_stop_end]
            test_idx = val_idx[split_point:]

            # 取出原始資料
            X_train_raw, y_train = X_raw[train_idx], y[train_idx]
            X_es_raw, y_es = X_raw[early_stop_idx], y[early_stop_idx]
            X_test_raw, y_test = X_raw[test_idx], y[test_idx]

            scaler = RobustScaler()

            # 將 3D (Batch, TimeSteps, Features) 壓平為 2D 讓 Scaler 學習
            X_train_2d = X_train_raw.reshape(-1, num_features)
            scaler.fit(X_train_2d)

            # 轉換並膨脹回原來的 3D 形狀
            X_train = scaler.transform(X_train_2d).reshape(X_train_raw.shape)
            X_es = scaler.transform(X_es_raw.reshape(-1, num_features)).reshape(X_es_raw.shape)
            X_test = scaler.transform(X_test_raw.reshape(-1, num_features)).reshape(X_test_raw.shape)

            train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
            es_loader = self._create_dataloader(X_es, y_es, shuffle=False)
            test_loader = self._create_dataloader(X_test, y_test, shuffle=False)

            pos_weight_val = MLTool.calculate_scale_weight(y_train)
            pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

            model = DLModelFactory.create(
                model_type=self.dl_model_type,
                num_features=num_features,
                time_steps=DLHyperParams.TIME_STEPS,
                rnn_type=self.rnn_type
            ).to(self.device)

            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=DLHyperParams.SCHEDULER_FACTOR, patience=DLHyperParams.SCHEDULER_PATIENCE
            )

            best_val_loss = float('inf')
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch_for_fold = 0

            # --- Training Loop ---
            for epoch in range(self.epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(X_batch), y_batch)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                # --- Validation (Early Stopping) ---
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    # 只看 es_loader (提早停止驗證集)
                    for X_v, y_v in es_loader:
                        X_v, y_v = X_v.to(self.device), y_v.to(self.device)
                        val_loss += criterion(model(X_v), y_v).item()

                avg_val_loss = val_loss / len(es_loader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    best_epoch_for_fold = epoch + 1 # 紀錄最佳 Epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= TrainConfig.EARLY_STOP_ROUND:
                        break

                scheduler.step(avg_val_loss)

            best_epochs.append(best_epoch_for_fold)

            # --- 收集 OOF 預測 (只針對完全沒看過的 test_loader) ---
            model.load_state_dict(best_model_wts)
            model.eval()
            test_preds = []
            with torch.no_grad():
                for X_t, _ in test_loader:
                    preds = torch.sigmoid(model(X_t.to(self.device))).cpu().numpy()
                    test_preds.extend(np.atleast_1d(preds))

            test_preds = np.array(test_preds)
            oof_predictions.iloc[test_idx] = test_preds

            # --- 計算指標 (用 y_test 對答案) ---
            y_pred_binary = (test_preds > 0.5).astype(int)
            acc = accuracy_score(y_test, y_pred_binary)
            cv_accuracies.append(acc)

            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, test_preds)
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

        dbg.log(f"【DL 驗證結果】平均 Accuracy: {np.mean(cv_accuracies):.4f}, 平均 AUC: {np.mean(cv_aucs):.4f}")
        return oof_predictions.dropna()

    def train_and_save_final_model(self, X_raw: np.ndarray, y: np.ndarray, save_path: Path | str):
        dbg.log(f"開始訓練最終上線版 DL 模型 (動態 Epoch={self.optimal_epochs})...")
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
            time_steps=DLHyperParams.TIME_STEPS,
            rnn_type=self.rnn_type
        ).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        model.train()
        for epoch in range(self.optimal_epochs):
            for X_batch, y_batch in full_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), str(save_path_obj))

        dbg.log(f"最終模型權重已儲存至: {save_path_obj}")
        return final_scaler

    def _create_dataloader(self, X: np.ndarray, y: np.ndarray | None = None, shuffle: bool = False) -> DataLoader:
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            y_tensor = torch.tensor(y, dtype=torch.float32)
            dataset = TensorDataset(X_tensor, y_tensor)
        else:
            dataset = TensorDataset(X_tensor)

        use_pin_memory = (self.device.type == 'cuda')
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=use_pin_memory
        )

    def load_inference_model(self, num_features: int, model_path: Path | str) -> nn.Module:
        """【供 UI 推論端使用】載入訓練好的模型權重"""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                dbg.error(f"❌ 深度學習模型載入失敗: 找不到檔案 {model_path}")
                return None

            model = DLModelFactory.create(
                model_type=self.dl_model_type,
                num_features=num_features,
                time_steps=DLHyperParams.TIME_STEPS,
                rnn_type=self.rnn_type
            ).to(self.device)

            model.load_state_dict(torch.load(str(model_path), map_location=self.device, weights_only=True))

            model.eval()
            dbg.log(f"✅ 成功載入 DL 模型: {model_path}")
            return model

        except RuntimeError as re:
            error_details = traceback.format_exc()
            dbg.error(f"💀 DL 模型結構不匹配 (Shape Mismatch)！\n您可能修改了特徵數量 (num_features={num_features}) 或 Time Steps，導致舊的權重檔塞不進去。\n請至 UI 介面點擊「強制深度重訓」！\n追蹤:\n{error_details}")
            return None

        except Exception as e:
            error_details = traceback.format_exc()
            dbg.error(f"🔥 DL 模型載入發生深層崩潰！\n目標路徑: {model_path}\n詳細錯誤追蹤:\n{error_details}")
            return None
