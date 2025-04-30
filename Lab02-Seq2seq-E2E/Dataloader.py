# import config
NAME_TOKEN = '[NAME]'
NEAR_TOKEN = '[NEAR]'
PAD_ID = 0

class Tokenizer: 
    """ 构建词到编号的映射编码器 Tokenizer: 定义一个 Tokenlizer 编码器，将单词映射到编号并构建词典：
    Encode 方法将给定字符串映射为数字编号，对于字符串 token_string，遍历每个词，调用 token2id 得到对应的编号列表。另外，在编号序列头尾分别加上开始和结束的特殊词 [BOS] 和 [EOS]，对未收录的词，标记为 [UNK]；
    Decode 方法将给定数字编号序列映射为字符串，对于编号列表 id_string，遍历每个编号，调用 id2token 得到对应的 token 列表。另外需要去掉 [BOS] 和 [EOS]，使用 join 返回解码的字符串。

        token_dict: 输入词到编号的映射词典
        _token_dict_rev: 反转键值对,方便查找编号对应的词
        vocab_size: 词典大小
    """
    def __init__(self, token_dict):
        self.token_dict = token_dict 
        self.token_dict_rev = {value: key for key, value in self.token_dict.items()} 
        self.vocab_size = len(self.token_dict) 
    
    """ 字符串编码: 将token映射为对应id,并在头尾加上特殊词[BOS]和[EOS]
        [BOS]: begin of string 标记字符串开头
        [EOS]: end of string 标记字符串结尾
        [UNK]: unknow token 词汇表中未收录的低频词
        [PAD]: padding 填充标记

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
        return self.token_dict_rev[token_id]
    
    def encode(self, token_string:list) -> list: 
        id_string = [self.token2id('[BOS]')]
        for _token in token_string:
            id_string.append(self.token2id(_token)) 
        id_string.append(self.token2id('[EOS]')) 
        return id_string
    
    def decode(self, id_string:list) -> str: 
        spec_tokens = {'[BOS]', '[EOS]', '[PAD]'}  
        token_string = [] 
        for _token_id in id_string:
            _token = self.id2token(_token_id) 
            if _token in spec_tokens: continue 
            token_string.append(_token) 
        # return token_string
        return ' '.join(token_string) 


import pandas as pd
import numpy as np
from re import split as re_split
from torch.utils.data import Dataset
from collections import Counter
""" 自定义 Dataloader (继承自 pytorch 的 Dataset 类): 本实验所使用的 E2E 数据集包含 trainset、devset、testset 三个 csv 文件，样本比例为 8:1:1，无需再做数据集划分。train/dev 数据集包括两个字段：结构化文本 mr 和参考文本 ref，test 数据集只有一个 MR 字段。首先需要用 pandas 读取数据集，由于编写代码时采用了虚拟环境，因此需要用 os 转换一个绝对路径。

    __init__: 初始化构造函数
    __getitem__: 
    __len__:
    create_field: 构造属性词典
    preprocess: 文本预处理
    create_voc: 构造文本词典
    sequence_padding: 序列填充和截断
"""
class E2EDataset(Dataset):    
    """ init 初始化: 根据 path 和 mode 加载数据集并赋值
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
            path = os.path.abspath(os.path.join(os.getcwd(), '.')) + "/e2e_dataset"
            if mode == 'train': path = path + "/trainset.csv"
            if mode == 'dev': path = path + "/devset.csv"
            if mode == 'test': path = path + "/testset.csv"
        
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
                raise ValueError("failed tokenizer and tokenizer must not be None") 
            self.field_tokenizer = field_tokenizer 
            self.key_num = len(self.field_tokenizer) 
            self.tokenizer = tokenizer 
            self.preprocess() 

        elif mode == 'test':
            _df = pd.read_csv(path)
            self.mr = self.str2dict(_df['MR'].values.tolist())   
            self.ref = ['' for _ in range(len(self.mr))]

            if field_tokenizer is None or tokenizer is None: 
                raise ValueError("failed tokenizer and tokenizer must not be None") 
            self.field_tokenizer = field_tokenizer 
            self.key_num = len(self.field_tokenizer) 
            self.tokenizer = tokenizer 
            self.preprocess() 

        else: 
            raise ValueError("Wrong input of mode='train|dev|test'")
        
    """ 将字符串转换为结构化的字典: 
        str_list: 传入 mr 字符串格式的结构化文本列表 (_df['MR'].values.tolist())
        _map_string: 利用分隔符分割结构化文本中的键值对,利用"["提取键和值并构建字典 -> _map_objs = _key + _value -> _dict
        dict_list: 返回处理后得到的字典列表
    """
    def str2dict(self, str_list:list) -> list: 
        dict_list = []

        _map_string = list(map(lambda x: x.split(', '), str_list))   
        
        for _map_obj in _map_string:  # ['A[a]', 'B[b]', ...] 
            _dict = {}
            for _item in _map_obj:
                _key = _item.split('[')[0]
                _value = _item.split('[')[1].replace(']', '')
                _dict[_key] = _value
            
            dict_list.append(_dict)
        
        return dict_list 
    
    """ 构造属性词典: 得到字典形式的格式化文本后，利用 collections 的 Counter 类统计所有键的出现次数，根据词频对属性名表进行排序，从 0 开始编号，最后 zip 组合为映射词典。

        mr_key: 提取 str2dict 处理 mr 字段得到的字典列表中的键,并调用Counter()统计词频,对属性名列表进行排序
        counter: from collections import Counter()
        _tokens: 保留排序后的词列表，创建属性名到编号的映射字典(从0开始编号)
        _key_num: 记录属性名的数量
    """
    def create_field(self) -> None: 
        mr_key = list(map(lambda x: list(x.keys()), self.mr)) 

        counter = Counter() 
        for line in mr_key: 
            counter.update(line) 
        _tokens = [(token, count) for token, count in counter.items()] 
        _tokens = sorted(_tokens, key=lambda x: -x[1])
        _tokens = [token for token, count in _tokens] 

        # 创建词典 token->id映射关系 
        self.field_tokenizer = dict(zip(_tokens, range(len(_tokens)))) 
        self.key_num = len(self.field_tokenizer)
    
    """ 文本预处理,包括分词和去词化等操作: 数据预处理步骤包括分词、去词化和词形还原等。由于本实验所处理的是结构化文本，因此可以按照文本规则进行分词，并去除标点符号：为了去除餐馆名称、地点等噪声词对模型训练的影响，可以在预处理时将 'name' 和 'near' 字段替换成占位符。由于结构化文本不存在语态问题，因此不需要做词形还原。

        raw_data_x: 结构化文本数据(特征)
        raw_data_y: 参考文本数据(目标值)
        lexicalizations: 去词化的原词(编解码后的句子)
        multi_data_y: 对应的多个参考文本(字典格式,键是结构化文本,值是参考文本的列表)

        mr_data:将结构化数据转化为长度为属性数量的列表
        PAD_ID: 填充词的编码,用以初始化结构化文本
        lex: 去词化原词(name,near)

        NAME_TOKEN: 对餐馆名name进行去词化,减少训练时的干扰
        NEAR_TOKEN: 对地点名near进行去词化,减少训练时的干扰
    """
    def preprocess(self) -> None: 
        self.raw_data_x = [] 
        self.raw_data_y = [] 
        self.lexicalizations = [] 
        self.muti_data_y = {}

        for _index in range(len(self.ref)): 
            # 将 mr 处理成列表并进行去词化 
            mr_data = [PAD_ID] * self.key_num 
            lex = ['', '']
            for _item in self.mr[_index].items(): 
                _key = _item[0] 
                _value = _item[1] 
                _key_idx = self.field_tokenizer[_key] 

                "去掉去词化步骤会对结果有影响吗? 对比实验一下:" 
                if _key == 'name': 
                    mr_data[_key_idx] = NAME_TOKEN 
                    lex[0] = _value
                elif _key == 'near': 
                    mr_data[_key_idx] = NEAR_TOKEN 
                    lex[1] = _value 
                else: 
                    mr_data[_key_idx] = _value
        
            # 将参考文本 ref 处理成列表并去词化
            ref_data = self.ref[_index] 
            if ref_data == '': ref_data = [''] # 如果是测试集(没有ref)则不做处理 
            else: 
                if lex[0]: ref_data = ref_data.replace(lex[0], NAME_TOKEN) 
                if lex[1]: ref_data = ref_data.replace(lex[1], NEAR_TOKEN) 
                ref_data = list(map(lambda x: re_split(r"([.,!?\"':;)(])", x)[0], ref_data.split())) 
                # 使用re的split去除数据中的标点符号，并按照空格分词
        
            self.raw_data_x.append(mr_data) 
            self.raw_data_y.append(ref_data) 
            self.lexicalizations.append(lex) 
            
            # 多参考文本 Convert all items to string before joining
            mr_data_str = ''.join(str(item) for item in mr_data)  
            if mr_data_str in self.muti_data_y.keys(): 
                self.muti_data_y[mr_data_str].append(self.ref[_index]) 
            else: self.muti_data_y[mr_data_str] = [self.ref[_index]] 

    """ 构造文本词典: 由于结构化文本中的属性值大多会出现在参考文本中，因此为了更好地反映结构化文本属性值和参考文本的关系，结构化文本属性值和参考文本使用同一本词典进行编解码。首先使用 Counter 对属性值的文本和参考文本的词进行统计，按词频排序后将特殊词加入到词表，从 0 开始编号并创建映射词典，使用该词典建立 Tokenlizer。
    
        counter: from collections import Counter()
        _tokens: 对属性值的文本和参考文本的词按照词频统计排序,并保留词列表(同属性词典)

        [PAD],[BOS],[EOS],[UNK] 特殊词表
        token_id_dict: 创建 token->id 的映射词典(从0开始编号),并用新词典重新建立分词器
    """
    def create_voc(self) -> None: 
        counter = Counter() 
        for line in self.raw_data_x: 
            counter.update(line) 
        for line in self.raw_data_y: 
            counter.update(line) 
        _tokens = [(token, count) for token, count in counter.items()] 
        _tokens = sorted(_tokens, key=lambda x: -x[1]) 
        _tokens = ['[PAD]', '[BOS]', '[EOS]', '[UNK]'] + [token for token, count in _tokens]

        # 创建 token->id 的映射词典,并用新词典重新建立分词器
        token_id_dict = dict(zip(_tokens, range(len(_tokens)))) 
        self.tokenizer = Tokenizer(token_id_dict) 

    """ 序列填充和截断,对长度不合法的句子进行归一化: 接下来对序列进行归一化处理，对于长度不足或超过预设的句子，需要进行填充或截断。填充词 padding 默认为 None，若不输入则采用填充词 [PAD] 的 ID=0 进行填充。

        string: 输入句子
        max_len: 最大长度
        padding: 填充词,默认为[PAD]的id
        _padding_length: 预期填充的长度,若>0则需要填充,若<0则需要截断
    """
    def sequence_padding(self, string:str, max_len:int, padding=None) -> str: 
        if padding is None: 
            padding = self.tokenizer.token2id('[PAD]') 
        self.padding = padding 

        _padding_length = max_len - len(string) 
        if _padding_length > 0: 
            res = string + [padding] * _padding_length  
        else: res = string[:max_len] 
        return res
    
    """ 重写 __getitem__ 函数: __getitem__ 接受序号 index，得到结构化文本和目标参考文本，使用词典进行编码并进行序列填充得到 x 和 y。对于 train mode，直接 return x, y；对于 dev/test mode，需要从 lexicalizations 列表中读取当前文本中被去词化的词 lex 以及从 muti_data_y 字典中读取当前结构化文本的多参考文本列表 muti_y，并返回四个参数。
    
        index: 读取数据的输入序号,得到结构化文本和目标参考文本
        x,y: 经过编码并填充后得到的文本
        lex,muti_y: 如果是mode==dev|test,则要返回当前文本去词化后的词 lex 以及多参考文本列表 muti_data_y
    """
    
    def __getitem__(self, index:int) -> tuple: 
        x = np.array(self.sequence_padding(self.tokenizer.encode(self.raw_data_x[index]), self.max_src_len)) 
        y = np.array(self.sequence_padding(self.tokenizer.encode(self.raw_data_y[index]), self.max_tgt_len)) 

        if self.mode == 'train': return x, y 
        else: 
            lex = self.lexicalizations[index] 
            muti_y = self.muti_data_y[''.join(str(item) for item in self.raw_data_x[index])] 
            return x, y, lex, muti_y

    """ 重写 __len__ 函数,返回数据集的大小
    """
    def __len__(self) -> int: 
        return len(self.ref) 