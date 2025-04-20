NAME_TOKEN = '[NAME]'
NEAR_TOKEN = '[NEAR]'

BOS_ID = -1
EOS_ID = -2
PAD_ID = 0

from torch import device as torch_device
class Config():
    def __init__(self):
        self.train_data = './e2e_dataset/trainset.csv'
        self.dev_data = './e2e_dataset/devset.csv'
        self.test_data = './e2e_dataset/testset.csv'

        self.model_save_path = './model.pkl'
        self.result_save_path = './result.txt'

        self.max_src_len = 80   # 最大结构文本长度
        self.max_tgt_len = 80   # 最大参考文本长度
        
        self.embedding_dim = 256    # 词嵌入维度
        self.embedding_dropout = 0.1    # 词嵌入 Dropout
        self.encoder_input_size = 256   # 编码器输入维度
        self.encoder_hidden_size = 512  # 编码器隐藏单元数
        self.decoder_input_size = 512   # 解码器输入维度
        self.decoder_hidden_size = 512  # 解码器隐藏单元数

        self.n_epochs = 30  # 训练迭代次数
        self.val_num = 1    # 进行验证的代数
        self.batch_size = 16    # 批数据大小
        self.learning_rate = 0.1    # 学习率
        device = torch_device('cpu')    # 训练设备
