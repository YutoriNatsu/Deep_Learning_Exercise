import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, Trainer, TrainingArguments
from loralib import add_lora, mark_only_lora_as_trainable

# 自定义数据集类
class NewsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                text = data['text']
                entities = data['entities']
                
                # 将实体转换为标签序列
                labels = [-100] * self.max_length  # -100 表示忽略
                tokenized_input = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
                input_ids = tokenized_input['input_ids'][0]
                attention_mask = tokenized_input['attention_mask'][0]
                
                # 假设实体标签从0开始编号
                label_map = {'标题': 0, '时间': 1, '地点': 2, '人物': 3, '组织': 4, '事件': 5}
                for entity_type, entity_value in entities.items():
                    start_idx = text.find(entity_value)
                    if start_idx != -1:
                        end_idx = start_idx + len(entity_value)
                        encoded_start = self.tokenizer.encode_plus(text[:start_idx], add_special_tokens=False)['input_ids'][-1]
                        encoded_end = self.tokenizer.encode_plus(text[:end_idx], add_special_tokens=False)['input_ids'][-1]
                        
                        if encoded_start < self.max_length and encoded_end < self.max_length:
                            labels[encoded_start] = label_map[entity_type]
                            labels[encoded_start+1:encoded_end+1] = [label_map[entity_type]] * (encoded_end - encoded_start)
                
                self.examples.append({
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': torch.tensor(labels)
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

# 加载数据集
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_dataset = NewsDataset('news_dataset.jsonl', tokenizer)

# 加载模型
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=6)  # 假设有6个实体类型

# 应用LoRA
add_lora(model.bert.encoder.layer, r=8)  # 添加LoRA适配器，秩为8
mark_only_lora_as_trainable(model)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# 开始微调
trainer.train()

# 保存模型
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')