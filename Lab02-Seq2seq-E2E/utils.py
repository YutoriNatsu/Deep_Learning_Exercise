import torch
import Dataloader as _dl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
print("using", device, "...") 

# 读取模型 
model = torch.load('./model.pkl').to(device) 
dataset = _dl.E2EDataset('./e2e_dataset/trainset.csv', mode='train') 
dataset.mode = 'dev' 

# 获得数据 
src, tgt, lex, _ = dataset[0] 
src = torch.as_tensor(src[np.newaxis, :]).to(device).transpose(0, 1) 
sentence, attention = model.predict(src)

# 还原文本 
src_txt = list(map(lambda x: dataset.tokenizer.id_to_token(x), src.flatten().cpu().numpy().tolist()[:10])) 
for i in range(len(src_txt)): 
    if src_txt[i] == '[NAME]': 
        src_txt[i] = lex[0] 
    elif src_txt[i] == '[NEAR]': 
        src_txt[i] = lex[1] 

sentence_txt = list(map(lambda x: dataset.tokenizer.id_to_token(x), sentence)) 
for i in range(len(src_txt)): 
    if sentence_txt[i] == '[NAME]': 
        sentence_txt[i] = lex[0] 
    elif sentence_txt[i] == '[NEAR]': 
        sentence_txt[i] = lex[1] 

# 绘制热力图 
ax = sns.heatmap(np.array(attention)[:, :10] * 100, cmap='YlGnBu') 
# 设置坐标轴 
plt.yticks([i + 0.5 for i in range(len(sentence_txt))], labels=sentence_txt, rotation=360, fontsize=12) 
plt.xticks([i + 0.5 for i in range(len(src_txt))], labels=src_txt, fontsize=12) 
plt.show() 