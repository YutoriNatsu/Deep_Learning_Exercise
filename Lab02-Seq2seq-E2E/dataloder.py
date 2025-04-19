
class Tokenizer: 
    """ 构建词到编号的映射编码器 Tokenizer:
        token_dict: 输入词到编号的映射词典
        _token_dict_rev: 反转键值对,方便查找编号对应的词
        _token_dict_size: 词典大小
    """
    def __init__(self, token_dict):
        self.token_dict = token_dict 
        self._token_dict_rev = {value: key for key, value in self.token_dict.items()} 
        self._token_dict_size = len(self.token_dict) 
    
    """ 字符串编码: 将token映射为对应id,并在头尾加上特殊词[BOS]和[EOS]
        [BOS]: begin of string 标记字符串开头
        [EOS]: end of string 标记字符串结尾
        [UNK]: unknow token 词汇表中未收录的低频词

        token与id的映射:
            def token2id(self, token:str) -> int
            def id2token(self, token_id:int) -> str
        字符串的编解码:
            def encode(self, token_string:list[str]) -> list[int]
            def decode(self, id_string:list[int]) -> list[str]
        
    """
    def token2id(self, token:str) -> int:
        return self.token_dict.get(token, self.token_dict['[UNK]']) # 未找到时返回[UNK]
    def id2token(self, token_id:int) -> str: 
        return self._token_dict_rev[token_id]
    
    def encode(self, token_string:list[str]) -> list[int]: 
        id_string = [self.token2id('[BOS]')]
        for _token in token_string:
            id_string.append(self.token2id(_token)) 
        id_string.append(self.token2id('[EOS]')) 
        return id_string
    
    def decode(self, id_string:list[int]) -> list[str]: 
        spec_tokens = {'[BOS]', '[EOS]', '[PAD]'}  
        token_string = [] 
        for _token_id in id_string:
            _token = self.id2token(_token_id) 
            if _token in spec_tokens: continue 
            token_string.append(_token) 
        return token_string
        # return ' '.join(tokens) 


import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from collections import Counter
""" 自定义 Dataloader (继承自 pytorch 的 Dataset 类):
    
"""
class E2EDataset(Dataset):    
    """ init 初始化:
        path: 外部传入数据集的路径,如果为空则调用内部方法用os获取相对路径
        mode: 数据集类型
        field_tokenizer: 属性词典
        tokenizer: 文本词典
        max_src_len: 结构化文本的最大长度
        max_tgt_len: 目标参考文本的最大长度
    """
    def __init__(self, path="", mode='train',
                 field_tokenizer=None, tokenizer=None,
                 max_src_len=80, max_tgt_len=80): 
        self.mode = mode
        self.max_src_len = max_src_len 
        self.max_tgt_len = max_tgt_len

        if path == "": 
            # 由于编写代码时采用了虚拟环境, 因此需要用 os 转换一个绝对路径
            import os
            path = os.path.abspath(os.path.join(os.getcwd(), '.')) + "\\e2e_dataset"
            if mode == 'train': path = path + "\\trainset.csv"
            if mode == 'dev': path = path + "\\devset.csv"
            if mode == 'test': path = path + "\\testset.csv"
        
        if mode == 'train':
            _df = pd.read_csv(path) 
            self.mr = self.str2dict(_df['mr'].values.tolist())   
            self.ref = _df['ref'].values.tolist()

            self.create_field() 
            self.preprocess() 
            self.create_voc() 

        elif mode == 'dev':
            _df = pd.read_csv(path) 
            self.mr = self.str2dict(_df['mr'].values.tolist())   
            self.ref = _df['ref'].values.tolist()

            if field_tokenizer is None or tokenizer is None: 
                raise ValueError("field tokenizer and tokenizer must not be None") 
            self.field_tokenizer = field_tokenizer 
            self.key_num = len(self.field_tokenizer) 
            self.tokenizer = tokenizer 
            self.preprocess() 

        elif mode == 'test':
            _df = pd.read_csv(path)
            self.mr = self.str2dict(_df['MR'].values.tolist())   
            self.ref = ['' for _ in range(len(self.mr))]

            if field_tokenizer is None or tokenizer is None: 
                raise ValueError("field tokenizer and tokenizer must not be None") 
            self.field_tokenizer = field_tokenizer 
            self.key_num = len(self.field_tokenizer) 
            self.tokenizer = tokenizer 
            self.preprocess() 

        else: 
            raise ValueError("Wrong input of mode='train|dev|test'")
        
    """ 将字符串转换为结构化的字典: 
    
    """
    def str2dict(str_list): 
        dict_list = [] 
        # 利用分隔符逗号将属性和值对分开 
        map_os = list(map(lambda x: x.split(', '), str_list))   
        
        for map_o in map_os:  # ['A[a]', 'B[b]', ...] 
            _dict = {}
            for item in map_o:
                key = item.split('[')[0]
                value = item.split('[')[1].replace(']', '')
                dict[key] = value
            dict_list.append(_dict)
        
        return dict_list 
    
    """ 构造属性词典 field_tokenlizer:
    
    """
    def create_field(self): 
        # mr字段 值 
        mr_key = list(map(lambda x: list(x.keys()), self.mr)) 
        # 统计词频 
        counter = Counter() 
        for line in mr_key: 
            counter.update(line) 
        # 按词频排序 
        _tokens = [(token, count) for token, count in counter.items()] 
        _tokens = sorted(_tokens, key=lambda x: -x[1]) 
        # 去掉词频，只保留词列表 
        _tokens = [token for token, count in _tokens] 
        # 创建词典 token->id映射关系 
        self.field_tokenizer = dict(zip(_tokens, range(len(_tokens)))) 
        self.key_num = len(self.field_tokenizer)
    
    """ 文本预处理:
    
    """
    def preprocess(self): 
        self.raw_data_x = [] 
        self.raw_data_y = [] 
        self.lexicalizations = [] 
        self.muti_data_y = {} 
        for index in range(len(self.ref)): 
            mr_data = [PAD_ID] * self.key_num 
            lex = ['', '']
        
        # 将mr处理成列表并进行Delexicalization 
        for item in self.mr[index].items(): 
            key = item[0] 
            value = item[1] 
            key_idx = self.field_tokenizer[key] 
            # Delexicalization 
            if key == 'name': 
                mr_data[key_idx] = NAME_TOKEN 
                lex[0] = value
            elif key == 'near': 
                mr_data[key_idx] = NEAR_TOKEN 
                lex[1] = value 
            else: 
                mr_data[key_idx] = value
        
        # 将ref处理成列表 
        ref_data = self.ref[index] 
        if ref_data == '':  # 句子为空说明是测试集没有ref 
            ref_data = [''] 
        else: 
            # Delexicalize 
            if lex[0]: 
                ref_data = ref_data.replace(lex[0], NAME_TOKEN) 
            if lex[1]: 
                ref_data = ref_data.replace(lex[1], NEAR_TOKEN) 
            ref_data = list(map(lambda x: re.split(r"([.,!?\"':;)(])", x)[0], ref_data.split())) 
        
        # 追加列表 
        self.raw_data_x.append(mr_data) 
        self.raw_data_y.append(ref_data) 
        self.lexicalizations.append(lex) 
        # 多参考文本 
        mr_data_str = ''.join(mr_data) 
        if mr_data_str in self.muti_data_y.keys(): 
            self.muti_data_y[mr_data_str].append(self.ref[index]) 
        else: 
            self.muti_data_y[mr_data_str] = [self.ref[index]] 

    """ 文本词典构造函数:
    
    """
    def create_voc(self): 
        # 统计词频 
        counter = Counter() 
        for line in self.raw_data_x: 
            counter.update(line) 
        for line in self.raw_data_y: 
            counter.update(line) 
        # 按词频排序 
        _tokens = [(token, count) for token, count in counter.items()] 
        _tokens = sorted(_tokens, key=lambda x: -x[1]) 
        # 去掉词频，只保留词列表 
        _tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]'] + [token for token, count in _tokens] 
        # 创建词典 token->id映射关系 
        token_id_dict = dict(zip(_tokens, range(len(_tokens)))) 
        # 使用新词典重新建立分词器 
        self.tokenizer = Tokenizer(token_id_dict) 

    """ 序列填充和截断:
    
    """
    # 将给定数据填充到相同长度 
    def sequence_padding(self, data, max_len, padding=None): 
        # 计算填充数据 
        if padding is None: 
            padding = self.tokenizer.token_to_id('[PAD]') 
        self.padding = padding 
        # 开始填充 
        padding_length = max_len - len(data) 
        # 不足就进行填充
        if padding_length > 0: 
            outputs = data + [padding] * padding_length 
        # 超过就进行截断 
        else: 
            outputs = data[:max_len] 
        return outputs
    
    """ 重写 __getitem__ 函数
    
    """
    
    def __getitem__(self, index): 
        x = np.array(self.sequence_padding(self.tokenizer.encode(self.raw_data_x[index]), self.max_src_len)) 
        y = np.array(self.sequence_padding(self.tokenizer.encode(self.raw_data_y[index]), self.max_tgt_len)) 
        if self.mode == 'train': 
            return x, y 
        else: 
            lex = self.lexicalizations[index] 
            muti_y = self.muti_data_y[''.join(self.raw_data_x[index])] 
            return x, y, lex, muti_y

    """ 重写 __len__ 函数
    
    """
    # 使用len函数时调用 返回数据集大小 
    def __len__(self): 
        return len(self.ref) 