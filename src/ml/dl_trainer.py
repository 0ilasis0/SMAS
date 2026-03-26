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
from ml.params import DLHyperParams, RNNType, TrainConfig
from path import PathConfig


# -----------------------------------------
# 混合神經網路架構：1D-CNN 特徵萃取 + LSTM 趨勢記憶
# -----------------------------------------
class CNN_RNN(nn.Module):
    def __init__(
            self,
            num_features: int,
            rnn_type: RNNType = RNNType.LSTM,
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
        self.pool = nn.MaxPool1d(kernel_size=2)

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
    def __init__(self, ticker: str, rnn_type: RNNType = RNNType.LSTM):
        self.rnn_type = rnn_type
        self.ticker = ticker
        self.model_save_path = PathConfig.get_dl_model_path(self.ticker, self.rnn_type)

        self.device = self._detect_device()

        dbg.log(f"DLTrainer 初始化 [{self.ticker} - {self.rnn_type.name}]，存檔路徑: {self.model_save_path}")

        self.batch_size = DLHyperParams.BATCH_SIZE
        self.epochs = DLHyperParams.EPOCHS
        self.learning_rate = DLHyperParams.LEARNING_RATE

    def _detect_device(self):
        """自動偵測是否支援 GPU 加速 or Mac 晶片加速"""
        if torch.cuda.is_available(): return torch.device("cuda")
        if torch.backends.mps.is_available(): return torch.device("mps")
        return torch.device("cpu")

    def train_with_cv(self, X: np.ndarray, y: np.ndarray, original_index: pd.Index, n_splits: int = TrainConfig.N_SPLITS) -> pd.Series:
        """
        執行 TimeSeriesSplit 交叉驗證，並回傳 OOF 預測值給 Meta-Learner。
        """
        n_splits = MathTool.clamp(n_splits, TrainConfig.N_SPLITS_MIN, TrainConfig.N_SPLITS_MAX)
        dbg.log(f"開始執行 DL (CNN-LSTM) TimeSeriesSplit (Fold={n_splits})...")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof_predictions = pd.Series(index=original_index, dtype=float)

        cv_accuracies, cv_aucs = [], []
        num_features = X.shape[2]

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

            # 初始化模型與訓練工具
            model = CNN_RNN(num_features=num_features, rnn_type=self.rnn_type).to(self.device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=DLHyperParams.SCHEDULER_FACTOR,
                patience=DLHyperParams.SCHEDULER_PATIENCE
            )

            best_val_loss = float('inf')
            patience_counter = 0

            # --- Training Loop ---
            for epoch in range(self.epochs):
                model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()
                    loss = criterion(model(X_batch), y_batch)
                    loss.backward()
                    optimizer.step()

                # --- Validation & Early Stopping ---
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for X_v, y_v in val_loader:
                        X_v, y_v = X_v.to(self.device), y_v.to(self.device)
                        val_loss += criterion(model(X_v), y_v).item()

                avg_val_loss = val_loss / len(val_loader)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= TrainConfig.EARLY_STOP_ROUND:
                        break # 觸發 Early Stopping

                scheduler.step(avg_val_loss)

            # --- 收集 OOF 預測 ---
            model.eval()
            val_preds = []
            with torch.no_grad():
                for X_v, _ in val_loader:
                    # 推論後搬回 CPU 並轉成 numpy
                    preds = torch.sigmoid(model(X_v.to(self.device))).cpu().numpy()
                    val_preds.extend(np.atleast_1d(preds))

            val_preds = np.array(val_preds)
            oof_predictions.iloc[val_idx] = val_preds

            # --- 計算指標 ---
            y_pred_binary = (val_preds > 0.5).astype(int)
            acc = accuracy_score(y_val, y_pred_binary)
            cv_accuracies.append(acc)

            if len(np.unique(y_val)) > 1:
                auc = roc_auc_score(y_val, val_preds)
                cv_aucs.append(auc)
                auc_str = f"{auc:.4f}"
            else:
                auc_str = "N/A"

            dbg.log(f"Fold {fold+1}: Accuracy = {acc:.4f}, AUC = {auc_str}")

        if not cv_accuracies:
            dbg.error("交叉驗證失敗：資料量不足。")
            return pd.Series(dtype=float)

        dbg.log(f"【DL 驗證結果】平均 Accuracy: {np.mean(cv_accuracies):.4f}, 平均 AUC: {np.mean(cv_aucs):.4f}")
        return oof_predictions.dropna()

    def train_and_save_final_model(self, X: np.ndarray, y: np.ndarray):
        """使用全量資料訓練最終上線版本"""
        dbg.log("開始訓練最終上線版 DL 模型 (使用全量資料)...")
        num_features = X.shape[2]

        full_loader = self._create_dataloader(X, y, shuffle=True)
        model = CNN_RNN(num_features=num_features, rnn_type=self.rnn_type).to(self.device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in full_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.model_save_path)
        dbg.log(f"最終模型權重已儲存至: {self.model_save_path}")

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

    def load_inference_model(self, num_features: int) -> nn.Module:
        """【供 UI 推論端使用】載入訓練好的模型權重"""
        try:
            model = CNN_RNN(num_features=num_features, rnn_type=self.rnn_type).to(self.device)
            model.load_state_dict(torch.load(self.model_save_path, map_location=self.device, weights_only=True))
            model.eval() # 記得切換到推論模式
            dbg.log(f"成功載入 DL 模型: {self.model_save_path}")
            return model
        except Exception as e:
            dbg.error(f"模型載入失敗: {e}")
            return None
