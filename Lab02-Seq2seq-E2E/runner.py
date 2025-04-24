import BLEUscore as _bleu
import Dataloader as _dl

# import config
# cfg = config.Config()
import Model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sys import stdout
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time

""" 模型运行实例: 用于 main.ipynb
    device: 选择训练设备 (cpu 或 cuda)
    cfg: 加载 config.py 中的参数
    
    def train: 模型训练
    def evaluate: 模型验证 (使用 BLEU-score)
    def illust_trainning_curve: 绘制 lr/loss/bleu_score 曲线
    def illust_heatmap: 绘制 attention 热力图
    def do: 运行模型 (训练和验证)
    def test: 进行测试
"""
class Runner():
    def __init__(self, cfg, device='', showExample=False): 
        try:
            self.train_set = _dl.E2EDataset(mode='train', max_src_len=cfg.max_src_len, max_tgt_len=cfg.max_tgt_len)

            self.dev_set = _dl.E2EDataset(mode='dev', max_src_len=cfg.max_src_len, max_tgt_len=cfg.max_tgt_len, field_tokenizer=self.train_set.field_tokenizer, tokenizer=self.train_set.tokenizer) 

            self.test_set = _dl.E2EDataset(mode='test', max_src_len=cfg.max_src_len, max_tgt_len=cfg.max_tgt_len, field_tokenizer=self.train_set.field_tokenizer, tokenizer=self.train_set.tokenizer) 

        except ValueError as e: print("Error in Dataloader: ", repr(e))

        self.train_loader = DataLoader(self.train_set, batch_size=cfg.batch_size, shuffle=True) 

        self.model = Model.E2EModel(cfg, src_vocab_size=self.train_set.tokenizer.vocab_size, tgt_vocab_size=self.train_set.tokenizer.vocab_size).to(self.device)

        if (showExample==True):
            print("Example:")
            pass
        
        if len(device)==0: self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else: self.device = device

        # 初始化batch大小，epoch数，损失函数，优化器等 
        self.batch_size = cfg.batch_size 
        self.weight = torch.ones(self.train_set.tokenizer.vocab_size) 
        self.weight[cfg.PAD_ID] = 0

        # 定义分数评估 
        self.scorer = _bleu.BLEUScore(max_ngram=4) 

        # 定义损失函数 
        self.criterion = nn.NLLLoss(self.weight, size_average=True).to(self.device) 

        # 定义优化器 
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=cfg.learning_rate) 
        # self.optimizer = optim.Adam(params=self.model.parameters(), lr=cfg.learning_rate)

        # 定义学习率下降 
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200) 
        self.MAX_EPOCH = cfg.n_epochs 
        self.VAL_NUM = cfg.val_num 

        self.best_bleu = 0.0  # 最高bleu 
        self.loss_li = [] 
        self.bleu_li = [] 
        self.lr_li = []

    def train(self, iterator, iter) -> None: 
        _print_loss = 0.0
        self.model.train() 

        with tqdm(total=len(iterator), desc='epoch{} [train]'.format(iter), file=stdout) as t: 
            for i, batch in enumerate(iterator):
                src, tgt = batch # 移至设备上 
                src = src.to(self.device).transpose(0, 1) 
                tgt = tgt.to(self.device).transpose(0, 1)

                self.optimizer.zero_grad()  # 初始化梯度值 
                
                # Forward 
                logits = self.model((src, tgt)) 
                vocab_size = logits.size()[-1] 
                logits = logits.contiguous().view(-1, vocab_size) 
                targets = tgt.contiguous().view(-1, 1).squeeze(1) 
                loss = self.criterion(logits, targets.long()) 
                _print_loss += loss.data.item() 

                loss.backward() # Backward 
                self.optimizer.step() # Update
                loss.backward() # Backward 
                self.optimizer.step() # Update

                t.set_postfix(loss=_print_loss / (i + 1), lr=self.scheduler.get_last_lr()[0]) 
                t.update(1) 

                self.loss_li.append(_print_loss / len(iterator)) 
                self.lr_li.append(self.scheduler.get_last_lr()[0]) 
                self.scheduler.step() 

    def evaluate(self, iterator, iter, save_path:str="") -> None: 
        self.model.eval() 
        bleu = 0.0 
        total_num = 0 
        # 重置分数统计器 
        self.scorer.reset() 
        with torch.no_grad(): 
            for data in tqdm(iterator, desc='{} [valid]'.format(" " * (5 + len(str(iter)))), file=stdout): 
                # 重置分数统计器 
                src, tgt, lex, muti_tgt = data 
                src = torch.as_tensor(src[:, np.newaxis]).to(self.device) 
                sentence, attention = self.model.predict(src) 
                # 解码句子 
                sentence = self.train_set.tokenizer.decode(sentence).replace('[NAME]', lex[0]).replace('[NEAR]', lex[1]) 
                self.scorer.append(sentence, muti_tgt) 
            
            bleu = self.scorer.score()
            self.bleu_li.append(bleu) 
            # print("BLEU SCORE: {:.4f}".format(bleu)) 
            if bleu > self.best_bleu: 
                self.best_bleu = bleu 
                
                if len(save_path)!=0:
                    torch.save(self.model, save_path) 
                    # print("model saved.") 

    def illust_trainning_curve(self, saveFig:str="") -> None:
        train_x = []
        valid_x = []
        for i in range(1,self.MAX_EPOCH+1):
            train_x.append(i)
            if i % self.VAL_NUM == 0: valid_x.append(i)

        plt.figure(figsize=(20,10))
        ax=plt.axes()
        ax.plot(train_x, self.loss_li, marker='x', linestyle = ':', lw=2, label=r"train_loss")
        ax.plot(train_x, self.lr_li, marker='s', linestyle='--', lw=2, label=r"train_lr")
        ax.plot(valid_x, self.bleu_li, marker='o', linestyle = '-', lw=2, label=r"valid_bleu")
        
        plt.xlabel('epoch', fontsize=12)
        plt.title("trainning process", fontsize=16)

        if len(saveFig)!=0: plt.savefig(saveFig)
        plt.show()
        return None

    def illust_heatmap(self, saveFig:str="") -> None:
        # 获得数据 
        _src, _tgt, _lex, _ = self.train_set[0] 
        _src = torch.as_tensor(_src[np.newaxis, :]).to(self.device).transpose(0, 1) 
        sentence, attention = self.model.predict(_src)

        # 还原文本 
        src_txt = list(map(lambda x: self.train_set.tokenizer.id_to_token(x), _src.flatten().cpu().numpy().tolist()[:10])) 
        for i in range(len(src_txt)): 
            if src_txt[i] == '[NAME]': src_txt[i] = _lex[0] 
            elif src_txt[i] == '[NEAR]': src_txt[i] = _lex[1] 

        sentence_txt = list(map(lambda x: self.train_set.tokenizer.id_to_token(x), sentence)) 
        for i in range(len(src_txt)): 
            if sentence_txt[i] == '[NAME]': sentence_txt[i] = _lex[0] 
            elif sentence_txt[i] == '[NEAR]': sentence_txt[i] = _lex[1] 

        # draw heatmap
        ax = sns.heatmap(np.array(attention)[:, :10] * 100, cmap='YlGnBu') 
        plt.yticks([i + 0.5 for i in range(len(sentence_txt))], labels=sentence_txt, rotation=360, fontsize=12) 
        plt.xticks([i + 0.5 for i in range(len(src_txt))], labels=src_txt, fontsize=12) 
        
        if len(saveFig)!=0: plt.savefig(saveFig)
        plt.show() 
        return None

    def do(self, save_path:str="", output_log = False) -> None:
        try:
            _begin = time.time()
            for epoch in range(self.MAX_EPOCH): 
                self.train(self.train_loader, epoch)

                if epoch % self.VAL_NUM == 0: 
                    _end = time.time()

                    self.evaluate(self.dev_set, epoch, save_path) 

                    if output_log==True:
                        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] epoch {epoch+1} lr:{self.lr_li[-1]:.4f} | loss:{self.loss_li[-1]:.4f} | bleu_score:{self.bleu_li[-1]:.4f} | time:{(_end-_begin):.2f}s")
                    _begin = time.time()
            
            print("train success!")
        except:
            print("something wrong with trainning process")
        return None

    def test(self, save_path:str="", output_log=False) -> None:
        self.model.eval() 
        with torch.no_grad(): 
            for data in tqdm(self.test_set, desc='[test]', file=stdout): 
                src, tgt, lex, _ = data 
                src = torch.as_tensor(src[:, np.newaxis]).to(self.device) 
               
                sentence, attention = self.model.predict(src) 
                sentence = self.train_set.tokenizer.decode(sentence).replace('[NAME]', lex[0]).replace('[NEAR]', lex[1]) 
                
                if output_log==True: print(sentence)
                if len(save_path)!=0:
                    with open(save_path, 'a+', encoding='utf-8') as f: 
                        f.write(sentence + '.\n') 
        
        print('Finished Testing!')
        return None