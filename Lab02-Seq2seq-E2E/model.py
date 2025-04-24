# import config
BOS_ID = -1
EOS_ID = -2
PAD_ID = 0

import torch
import torch.nn as nn
from torch.autograd import Variable

""" 编码器 Encoder (MLP):
    input: 输入大小 input_size, 输出大小 hidden_size
    网络结构: 全连接层 Linear + ReLU 激活函数
"""
class Encoder(nn.Module): 
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__() 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.W = nn.Linear(self.input_size, self.hidden_size) 
        self.relu = nn.ReLU()
    
    def forward(self, input_embedded): 
        seq_len, batch_size, emb_dim = input_embedded.size() 
        outputs = self.relu(self.W(input_embedded.view(-1, emb_dim))) 
        outputs = outputs.view(seq_len, batch_size, -1) 
        dec_hidden = torch.sum(outputs, 0) 
        return outputs, dec_hidden.unsqueeze(0)

""" 解码器 Decoder (RNN-GRU):
    input: RNN 输入大小 input_size, 隐藏大小 hidden_size, 最终解码输出大小 output_size, 词嵌入维度 embedding_dim, 编码器的输出大小 encoder_hidden_size
    网络结构: 门控循环神经网络 nn.GRU + 自定义注意力模块 Attention -> 注意力权重与输出向量加权组合
"""
class Decoder(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size, embedding_dim, encoder_hidden_size): 
        super(Decoder, self).__init__() 
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=False) 
        self.attn_module = Attention(encoder_hidden_size, hidden_size) 
        self.W_combine = nn.Linear(embedding_dim + encoder_hidden_size, hidden_size) 
        self.W_out = nn.Linear(hidden_size, output_size) 
        self.log_softmax = nn.LogSoftmax() 

    def forward(self, prev_y_batch, prev_h_batch, encoder_outputs_batch): 
        attn_weights = self.attn_module(prev_h_batch, encoder_outputs_batch)  # B x SL 
        # 对编码器的输出进行加权 
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs_batch.transpose(0, 1))  # B x 1 x MLP_H 
        # 经过RNN（GRU）解码 
        # B x (prev_y_dim+(enc_dim * num_enc_directions)) 
        y_ctx = torch.cat((prev_y_batch, context.squeeze(1)), 1)   
        rnn_input = self.W_combine(y_ctx)  # B x H 
        dec_rnn_output, dec_hidden = self.rnn(rnn_input.unsqueeze(0), prev_h_batch)  # 1 x B x H, 1 x B x H 
        # 计算输出概率 
        unnormalized_logits = self.W_out(dec_rnn_output[0])  # B x TV 
        dec_output = self.log_softmax(unnormalized_logits)  # B x TV 
        # 返回最终输出、隐藏状态以及注意力权重 
        return dec_output, dec_hidden, attn_weights
    
""" 注意力模块 Attention:
    计算公式(Bahdanau Attention): a(s_{i-1},h_j)=v_a^T\tanh(W_as_{i-1}+U_ah_j)
"""
class Attention(nn.Module): 
    def __init__(self, enc_dim, dec_dim, attn_dim=None): 
        super(Attention, self).__init__() 
        self.num_directions = 1 
        self.h_dim = enc_dim 
        self.s_dim = dec_dim 
        self.a_dim = self.s_dim if attn_dim is None else attn_dim 
        # 构建注意力 
        self.U = nn.Linear(self.h_dim * self.num_directions, self.a_dim) 
        self.W = nn.Linear(self.s_dim, self.a_dim) 
        self.v = nn.Linear(self.a_dim, 1) 
        self.tanh = nn.Tanh() 
        self.softmax = nn.Softmax()
    
    def forward(self, prev_h_batch, enc_outputs): 
        src_seq_len, batch_size, enc_dim = enc_outputs.size() 
        uh = self.U(enc_outputs.view(-1, self.h_dim)).view(src_seq_len, batch_size, self.a_dim)  # SL x B x self.attn_dim 
        wq = self.W(prev_h_batch.view(-1, self.s_dim)).unsqueeze(0)  # 1 x B x self.a_dim 
        wq3d = wq.expand_as(uh) 
        wquh = self.tanh(wq3d + uh) 
        attn_unnorm_scores = self.v(wquh.view(-1, self.a_dim)).view(batch_size, src_seq_len) 
        attn_weights = self.softmax(attn_unnorm_scores)  # B x SL 
        return attn_weights 

""" Seq2Seq 模型:
    网络结构: Embedding + Dropout -> Encoder + Decoder
    predict: 模型推理 -> 在 decode 时对上一时刻的输出进行词嵌入然后进行解码得到输出、隐藏状态和注意力矩阵，选择最大的词概率对应的下标，即最大可能性的词编号，对词编号和注意力矩阵进行记录并将预测的当前时刻的词编号作为下一时刻的输入。最后，返回解码的结果和注意力矩阵。
"""
class E2EModel(nn.Module): 
    def __init__(self, cfg, src_vocab_size, tgt_vocab_size): 
        super(E2EModel, self).__init__() 
        self.device = cfg.device  # 设备 
        self.cfg = cfg 
        self.src_vocab_size = src_vocab_size 
        self.tgt_vocab_size = tgt_vocab_size 
        
        # 构建词嵌入层 
        self.embedding_mat = nn.Embedding(src_vocab_size, cfg.embedding_dim, padding_idx=PAD_ID) 
        self.embedding_dropout_layer = nn.Dropout(cfg.embedding_dropout) 
        # 构建编码器和解码器

        self.encoder = Encoder(input_size=cfg.encoder_input_size, 
                             hidden_size=cfg.encoder_hidden_size) 
        self.decoder = Decoder(input_size=cfg.decoder_input_size, 
                             hidden_size=cfg.decoder_hidden_size, 
                             output_size=tgt_vocab_size, 
                             embedding_dim=cfg.embedding_dim, 
                             encoder_hidden_size=cfg.encoder_hidden_size)
        
    def forward(self, data): 
        batch_x_var, batch_y_var = data  # SL x B, TL x B 
        # 词嵌入 # SL x B x E 
        encoder_input_embedded = self.embedding_mat(batch_x_var) 
        encoder_input_embedded = self.embedding_dropout_layer(encoder_input_embedded) 
        # 编码 SL x B x H; 1 x B x H 
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded) 
        # 解码 
        dec_len = batch_y_var.size()[0] 
        batch_size = batch_y_var.size()[1] 
        dec_hidden = encoder_hidden 
        dec_input = Variable(torch.LongTensor([BOS_ID] * batch_size)).to(self.device) 
        logits = Variable(torch.zeros(dec_len, batch_size, self.tgt_vocab_size)).to(self.device) 
        # 采用Teacher forcing机制，输入总是标准答案 
        for di in range(dec_len): # 上一输出的词嵌入，B x E 
            prev_y = self.embedding_mat(dec_input)               
            # 解码 
            dec_output, dec_hidden, attn_weights = self.decoder(prev_y, dec_hidden, encoder_outputs) 
            logits[di] = dec_output  # 记录输出词概率 
            dec_input = batch_y_var[di]  # 下一输入是标准答案 
        return logits
    
    def predict(self, input_var): 
        # 词嵌入 
        encoder_input_embedded = self.embedding_mat(input_var) 
        # 编码 
        encoder_outputs, encoder_hidden = self.encoder(encoder_input_embedded) 
        # 解码 
        dec_ids, attn_w = [], [] 
        curr_token_id = BOS_ID 
        curr_dec_idx = 0 
        dec_input_var = Variable(torch.LongTensor([curr_token_id])) 
        dec_input_var = dec_input_var.to(self.device) 
        dec_hidden = encoder_hidden[:1]  # 1 x B x enc_dim 
        # 直到EOS或达到最大长度 
        while curr_token_id != EOS_ID and curr_dec_idx <= self.cfg.max_tgt_len: 
            prev_y = self.embedding_mat(dec_input_var)  # 上一输出的词嵌入，B x E 
            # 解码 
            decoder_output, dec_hidden, decoder_attention = self.decoder(prev_y, dec_hidden, encoder_outputs) 
            # 记录注意力 
            attn_w.append(decoder_attention.data.cpu().numpy().tolist()[0])   
            topval, topidx = decoder_output.data.topk(1)   # 选择最大概率 
            curr_token_id = topidx[0][0] 
            # 记录解码结果 
            dec_ids.append(int(curr_token_id.cpu().numpy()))  
            # 下一输入 
            dec_input_var = (Variable(torch.LongTensor([curr_token_id]))).to(self.device)   
            curr_dec_idx += 1 
        return dec_ids, attn_w 
    