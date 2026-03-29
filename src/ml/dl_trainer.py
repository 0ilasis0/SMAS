# dl_trainer.py
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import DataLoader, TensorDataset

from base import MathTool
from debug import dbg
from ml.const import RNNType
from ml.params import DLHyperParams, TrainConfig


# -----------------------------------------
# 混合神經網路架構：1D-CNN 特徵萃取 + LSTM 趨勢記憶
# -----------------------------------------
class CNN_RNN(nn.Module):
    def __init__(
            self,
            num_features: int,
            rnn_type: RNNType,
            cnn_out_channels: int = DLHyperParams.CNN_OUT_CHANNELS,
            rnn_hidden: int = DLHyperParams.LSTM_HIDDEN
        ):
        super().__init__()
        self.rnn_type = rnn_type

        # --- CNN 區塊 (特徵降維去噪) ---
        # 輸入形狀預期: (Batch, num_features, Time_Steps)
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=cnn_out_channels,
            kernel_size=3,
            padding=1
        )
        # 批次正規化 (穩定特徵分佈)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.relu = nn.ReLU()
        # MaxPool 負責將時間軸長度減半
        self.pool = nn.MaxPool1d(kernel_size=DLHyperParams.KERNEL_SIZE)

        # --- RNN 區塊 (動態生成) ---
        # 輸入形狀預期: (Batch, 縮短後的Time_Steps, cnn_out_channels)
        if self.rnn_type == RNNType.LSTM:
            self.rnn = nn.LSTM(cnn_out_channels, rnn_hidden, num_layers=DLHyperParams.NUM_LAYERS, batch_first=True)
        else:
            self.rnn = nn.GRU(cnn_out_channels, rnn_hidden, num_layers=DLHyperParams.NUM_LAYERS, batch_first=True)

        # 手動加入 Dropout 防止全連接層死背數據
        self.dropout = nn.Dropout(p=DLHyperParams.DROPOUT)

        # --- 輸出層 ---
        self.fc = nn.Linear(rnn_hidden, 1)

    def forward(self, x):
        # 原始輸入 x: (Batch, Time_Steps, Features)
        # 轉換為 CNN 需要的形狀: (Batch, Features, Time_Steps)
        x = x.transpose(1, 2)

        x = self.conv1(x)     # -> (Batch, 32, Time_Steps)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)      # -> (Batch, 32, Time_Steps // 2)

        # 轉換為 LSTM 需要的形狀: (Batch, Time_Steps // 2, 32)
        x = x.transpose(1, 2)

        # LSTM 回傳 out, (hn, cn)；GRU 回傳 out, hn。所以用 out, _ 通吃！
        out, _ = self.rnn(x)

        # 取出最後一個時間步長的隱藏狀態 (代表吸收了所有歷史資訊的總結)
        last_time_step_out = out[:, -1, :] # -> (Batch, 64)
        last_time_step_out = self.dropout(last_time_step_out)

        # 輸出勝率
        logits = self.fc(last_time_step_out)
        return logits.squeeze(-1)


# -----------------------------------------
# DL 離線訓練器 (與 XGBTrainer API 對齊)
# -----------------------------------------
class DLTrainer:
    def __init__(self, ticker: str, rnn_type: RNNType):
        self.rnn_type = rnn_type
        self.ticker = ticker

        self.device = self._detect_device()

        dbg.log(f"DLTrainer 初始化 [{self.ticker} - {self.rnn_type.name}]")

        self.batch_size = DLHyperParams.BATCH_SIZE
        self.epochs = DLHyperParams.EPOCHS
        self.learning_rate = DLHyperParams.LEARNING_RATE

        self.optimal_epochs = self.epochs

    def _detect_device(self):
        """自動偵測是否支援 GPU 加速 or Mac 晶片加速"""
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def train_with_cv(self, X: np.ndarray, y: np.ndarray, original_index: pd.Index, lookahead: int, n_splits: int = TrainConfig.N_SPLITS) -> pd.Series:
        n_splits = MathTool.clamp(n_splits, TrainConfig.N_SPLITS_MIN, TrainConfig.N_SPLITS_MAX)
        dbg.log(f"開始執行 DL (CNN-LSTM) 嚴格三階段交叉驗證 (Fold={n_splits}, Gap={lookahead})...")

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=lookahead)
        oof_predictions = pd.Series(index=original_index, dtype=float)

        cv_accuracies, cv_aucs = [], []
        best_epochs = [] # 紀錄每個 Fold 的最佳停止點
        num_features = X.shape[2]

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            # 嚴格三階段切分 (Train, EarlyStop, Test)
            split_point = len(val_idx) // 2
            early_stop_end = split_point - lookahead

            if early_stop_end <= 0 or split_point >= len(val_idx):
                dbg.war(f"Fold {fold+1}: 樣本數不足以切割三階段，跳過。")
                continue

            early_stop_idx = val_idx[:early_stop_end]
            test_idx = val_idx[split_point:]

            X_train, y_train = X[train_idx], y[train_idx]
            X_es, y_es = X[early_stop_idx], y[early_stop_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            # 建立三組 DataLoader
            train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
            es_loader = self._create_dataloader(X_es, y_es, shuffle=False)
            test_loader = self._create_dataloader(X_test, y_test, shuffle=False)

            pos_count = y_train.sum()
            neg_count = len(y_train) - pos_count
            pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
            pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

            model = CNN_RNN(num_features=num_features, rnn_type=self.rnn_type).to(self.device)
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

    def train_and_save_final_model(self, X: np.ndarray, y: np.ndarray, save_path: Path | str):
        """使用全量資料訓練最終上線版本"""
        dbg.log(f"開始訓練最終上線版 DL 模型 (動態 Epoch={self.optimal_epochs})...")
        num_features = X.shape[2]
        full_loader = self._create_dataloader(X, y, shuffle=True)

        pos_count = y.sum()
        neg_count = len(y) - pos_count
        pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
        pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32).to(self.device)

        model = CNN_RNN(num_features=num_features, rnn_type=self.rnn_type).to(self.device)
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

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        dbg.log(f"最終模型權重已儲存至: {save_path}")

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
            model = CNN_RNN(num_features=num_features, rnn_type=self.rnn_type).to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            model.eval()
            dbg.log(f"成功載入 DL 模型: {model_path}")
            return model
        except Exception as e:
            dbg.error(f"模型載入失敗: {e}")
            return None
