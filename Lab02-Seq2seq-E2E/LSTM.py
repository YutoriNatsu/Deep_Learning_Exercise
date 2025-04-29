import torch
import torch.nn as nn
PAD_ID=0

class BiLSTM(nn.Module):
    def __init__(self, cfg, src_vocab_size, tgt_vocab_size): 
        super(BiLSTM, self).__init__()
        self.device = cfg.device
        self.cfg = cfg 
        self.src_vocab_size = src_vocab_size 
        self.tgt_vocab_size = tgt_vocab_size 
        
        # 构建词嵌入层 
        self.embedding_mat = nn.Embedding(src_vocab_size, cfg.embedding_dim, padding_idx=PAD_ID) 
        self.embedding_dropout_layer = nn.Dropout(cfg.embedding_dropout) 

        # 构建 Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=self.cfg.embedding_size,  # 输入大小为转化后的词向量
            hidden_size=self.cfg.hidden_size,  # 隐藏层大小
            num_layers=self.cfg.num_layers,  # 堆叠层数，有几层隐藏层就有几层
            dropout=self.cfg.dropout,  # 遗忘门参数
            bidirectional=True  # 双向LSTM
        )

        self.dropout = nn.Dropout(self.cfg.dropout)
        self.fc = nn.Linear(
            self.cfg.num_layers * self.cfg.hidden_size * 2,  # 因为双向所以要*2
            self.cfg.output_size
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        embedded = self.embeddings(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        feature = self.dropout(h_n)
        feature_map = torch.cat([feature[i, :, :] for i in range(feature.shape[0])], dim=-1)
        out = self.fc(feature_map)
        return self.softmax(out)
