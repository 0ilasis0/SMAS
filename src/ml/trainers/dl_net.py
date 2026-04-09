import torch.nn as nn

from ml.const import DLModelType, RNNType
from ml.params import DLHyperParams


class CNN_RNN(nn.Module):
    ''' 混合神經網路架構：1D-CNN 特徵萃取 + LSTM 趨勢記憶 '''
    def __init__(self, num_features: int, rnn_type: RNNType):
        super().__init__()
        self.rnn_type = rnn_type

        cnn_out_channels = DLHyperParams.CNN_OUT_CHANNELS
        rnn_hidden = DLHyperParams.LSTM_HIDDEN
        dropout_rate = DLHyperParams.DROPOUT
        kernel_size = DLHyperParams.KERNEL_SIZE
        num_layers = DLHyperParams.NUM_LAYERS

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
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)

        # --- RNN 區塊 (動態生成) ---
        # 輸入形狀預期: (Batch, 縮短後的Time_Steps, cnn_out_channels)
        if self.rnn_type == RNNType.LSTM:
            self.rnn = nn.LSTM(cnn_out_channels, rnn_hidden, num_layers=num_layers, batch_first=True)
        elif self.rnn_type == RNNType.GRU:
            self.rnn = nn.GRU(cnn_out_channels, rnn_hidden, num_layers=num_layers, batch_first=True)

        # 手動加入 Dropout 防止全連接層死背數據
        self.dropout = nn.Dropout(p=dropout_rate)

        # --- 輸出層 ---
        self.fc = nn.Linear(rnn_hidden, 1)

    def forward(self, x):
        # 原始輸入 x: (Batch, Time_Steps, Features)
        # 轉換為 CNN 需要的形狀: (Batch, Features, Time_Steps)
        x = x.transpose(1, 2)

        x = self.conv1(x)     # -> (Batch, cnn_out_channels, Time_Steps)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)      # -> (Batch, cnn_out_channels, Time_Steps // kernel_size)

        # 轉換為 LSTM 需要的形狀: (Batch, Time_Steps // kernel_size, cnn_out_channels)
        x = x.transpose(1, 2)

        # LSTM 回傳 out, (hn, cn)；GRU 回傳 out, hn。所以用 out, _ 通吃！
        out, _ = self.rnn(x)

        # 取出最後一個時間步長的隱藏狀態 (代表吸收了所有歷史資訊的總結)
        last_time_step_out = out[:, -1, :] # -> (Batch, rnn_hidden)
        last_time_step_out = self.dropout(last_time_step_out)

        # 輸出勝率
        logits = self.fc(last_time_step_out)
        return logits.squeeze(-1)


class PureCNN1D(nn.Module):
    ''' 極速輕量化純 1D-CNN '''
    def __init__(self, num_features: int, time_steps: int):
        super().__init__()

        cnn_out = DLHyperParams.CNN_OUT_CHANNELS
        dropout_rate = DLHyperParams.DROPOUT

        # 第一層卷積
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=cnn_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2) # time_steps 減半

        # 第二層卷積 (提取更深層特徵，通常通道數翻倍)
        self.conv2 = nn.Conv1d(in_channels=cnn_out, out_channels=cnn_out * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2) # time_steps 再減半

        # 計算攤平後的維度
        flattened_time_steps = time_steps // 4
        self.flattened_size = (cnn_out * 2) * flattened_time_steps

        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.flattened_size, 1)

    def forward(self, x):
        # x: (Batch, Time_Steps, Features) -> (Batch, Features, Time_Steps)
        x = x.transpose(1, 2)

        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))

        # 攤平
        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        logits = self.fc(x)
        return logits.squeeze(-1)


class DLModelFactory:
    ''' 兵工廠 (Model Factory) '''
    @staticmethod
    def create(model_type: DLModelType, num_features: int, time_steps: int = DLHyperParams.TIME_STEPS, rnn_type: RNNType = RNNType.LSTM) -> nn.Module:
        if model_type == DLModelType.HYBRID:
            return CNN_RNN(num_features=num_features, rnn_type=rnn_type)
        elif model_type == DLModelType.PURE_CNN:
            return PureCNN1D(num_features=num_features, time_steps=time_steps)
        else:
            raise ValueError(f"未支援的深度學習模型架構: {model_type}")
