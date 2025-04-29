""" 评价指标 BLEU-4 (参考来源: https://github.com/salesforce/WikiSQL): 本实验中使用的评价指标为 BLEU-4，该指标通过 unigram 评估候选文本是否正确使用了关键数据中的单词，通过高阶的 n-gram 评估句子是否流畅。此外，BLEU 还引入了参数BP (Brevity Penalt) 惩罚长度过短的候选文本。本实验评价指标参考 WikiSQL 中提供的 BLEU 代码，使其能够直接被主函数调用并在验证环节进行评价。

    __init__: 初始化最大 gram 数量和大小写敏感性
    append: 统计不同 gram 的命中数量和候选长度
    compute_hits: 计算命中次数并累加
    cand_lens: 统计每个 gram 的预测长度,并选择最相近的参考文本并记录参考文本的长度
"""
import math
from re import split as re_split
from collections import defaultdict
class BLEUScore: 
    TINY = 1e-15 
    SMALL = 1e-9 
    
    """ init 初始化:
        max_gram: 最大 n-gram 数量
        case_sensitive: 大小写敏感

        reset: 对命中列表和候选长度列表进行初始化 -> 初始化参考文本长度为 0, 以及两个用于记录不同 gram 的命中列表和候选长度列表
    """
    def __init__(self, max_ngram=4, case_sensitive=False): 
        self.max_ngram = max_ngram
        self.case_sensitive = case_sensitive 
        self.reset()
    
    def reset(self): 
        self.ref_len = 0 
        self.cand_lens = [0] * self.max_ngram 
        self.hits = [0] * self.max_ngram
    
    """ append 添加句子: 统计不同 gram 的命中数量和候选长度
        pred_sent: 要预测的句子
        ref_sents: 参考文本列表

        tokenize: 对输入的句子进行分词
    """
    def append(self, pred_sent, ref_sents): 
        pred_sent = pred_sent if isinstance(pred_sent, list) else self.tokenize(pred_sent) 
        ref_sents = [ref_sent if isinstance(ref_sent, list) else self.tokenize(ref_sent) for ref_sent in ref_sents]

        for i in range(self.max_ngram): 
            # 计算每个gram的命中次数 
            self.hits[i] += self.compute_hits(i + 1, pred_sent, ref_sents) 
            # 计算每个gram的预测长度 
            self.cand_lens[i] += len(pred_sent) - i 
        # 选择长度最相近的参考文本 
        closest_ref = min(ref_sents, key=lambda ref_sent: (abs(len(ref_sent) - len(pred_sent)), len(ref_sent))) 
        # 记录参考文本长度 
        self.ref_len += len(closest_ref)

    def tokenize(input_sent):
        return list(map(lambda x: re_split(r"([.,!?\"':;)(])", x)[0], input_sent.split()))
    
    """ 统计命中次数: 调用 get_ngram_counts 得到 ngram 的句子列表, 遍历句子列表统计预测句子和参考文本的命中次数
    """
    def compute_hits(self, n, pred_sent, ref_sents): 
        merged_ref_ngrams = self.get_ngram_counts(n, ref_sents) 
        pred_ngrams = self.get_ngram_counts(n, [pred_sent]) 
        hits = 0 
        for ngram, cnt in pred_ngrams.items(): 
            hits += min(merged_ref_ngrams.get(ngram, 0), cnt) 
        return hits 

    """ 得到按 n-gram 数进行聚合后的句子列表:
    """
    def get_ngram_counts(self, n, sents): 
        merged_ngrams = {} 
        # 按gram数聚合句子 
        for sent in sents: 
            ngrams = defaultdict(int) 
            if not self.case_sensitive: 
                ngrams_list = list(zip(*[[tok.lower() for tok in sent[i:]] for i in range(n)])) 
            else: 
                ngrams_list = list(zip(*[sent[i:] for i in range(n)])) 
            for ngram in ngrams_list: 
                ngrams[ngram] += 1 
            for ngram, cnt in ngrams.items(): 
                merged_ngrams[ngram] = max((merged_ngrams.get(ngram, 0),cnt)) 
        return merged_ngrams 
    
    """ 计算 BLEU score: 根据评分计算公式来处理
        1. 计算 BP 值 (分段函数)
        2. 计算 /sum log{P_n}, w_n 为 1/max_ngram
        3. 最后得到 BLEU = BP·exp(/sum w_n·log{P_n})
    """
    def score(self): 
        bp = 1.0 
        # c <= r : BP=e^(1-r/c) 
        # c > r : BP=1.0 
        if self.cand_lens[0] <= self.ref_len: 
            bp = math.exp(1.0 - self.ref_len / (float(self.cand_lens[0]) if self.cand_lens[0] else 1e-5)) 
        prec_log_sum = 0.0 
 
        for n_hits, n_len in zip(self.hits, self.cand_lens): 
            n_hits = max(float(n_hits), self.TINY)               
            n_len = max(float(n_len), self.SMALL) 
            # 计算∑logPn=∑log(n_hits/n_len) 
            prec_log_sum += math.log(n_hits / n_len) 
        return bp * math.exp((1.0 / self.max_ngram) * prec_log_sum) 

""" 调用示例: 参考课件给出的计算示例，本实验在 BLEUscore.py 复现了这一运算过程，经验证结果正确 (BELU-2 score=0.860264827803306)
    1. 实例化 BLEUScore 类, 设置 max_ngram=2
    2.  Candidate: the cat sat on the mat
        Reference : the cat is on the mat 
    3. 调用append 函数在实例中加入当前句子
    4. 计算 BLEU-2 score = 0.86
"""
if __name__ == '__main__': 
    scorer = BLEUScore(max_ngram=2) 
    sentence = 'the cat sat on the mat ' 
    target = ['the cat is on the mat '] 
    scorer.append(sentence, target) 
    print(scorer.score()) 