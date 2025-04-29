import pandas as pd
from datasets import Dataset
import torch.optim as optim
from transformers import BertTokenizer
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader 
import torch.nn as nn
import torch
from tqdm import tqdm
import os
from BERT import BertModel, BertForMaskedLM

# Load the pretrained tokenizer for "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print("Tokenizer vocabulary size:", len(tokenizer))  # Should print 30522

# Define the configuration dictionary for the model
config = {
    'vocab_size': len(tokenizer),  # Use ':' instead of '='
    'hidden_size': 768,
    'max_position_embeddings': 128,
    'num_attention_heads': 12,
    'num_hidden_layers': 12,
    'intermediate_size': 3072,
    'dropout': 0.1
}

# Initialize the model with the tokenizer's vocab size and specified architecture.
model = BertForMaskedLM(config)

# Move the model to GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 1. Data Loading and Tokenization
# Load each Parquet file into a DataFrame and store in a list.
path = os.path.abspath(os.path.join(os.getcwd(), '.')) + "\\e2e_dataset"
file_paths = [
    os.path.join(path, "trainset.csv"),
    os.path.join(path, "devset.csv"),
    os.path.join(path, "testset.csv")
]
dfs = [pd.read_parquet(fp) for fp in file_paths]

# Combine all DataFrames into a single DataFrame.
df = pd.concat(dfs, ignore_index=True)
print("DataFrame head:")
print(df.head())
dataset = Dataset.from_pandas(df)
print("Dataset preview:")
print(dataset)

# Define your tokenization function.
def tokenize_function(example):
    # Adjust the column name if needed; assumed here to be "text".
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# Tokenize the dataset.
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names,
    num_proc=16  # Adjust based on your CPU cores.
)

# Set format to PyTorch tensors.
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# 2. Create a DataLoader with Data Collator for MLM
# Data collator will dynamically mask tokens with the MLM probability.
batch_size = 32
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

dataloader = DataLoader(
    tokenized_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=data_collator  # This adds `labels` to the batch.
)

# 3. Model, Optimizer, and Training Setup
# Define the loss criterion.
# For MLM, we use CrossEntropyLoss and ignore the index -100 for unmasked tokens.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss(ignore_index=-100)

optimizer = optim.Adam(model.parameters(), lr=5e-5)
num_epochs = 10

print("Starting training...")
model.to(device)
model.train()
global_step = 0  # Counter for global training steps

""" Assuming the following objects are defined elsewhere:
  - model: your custom BERT model
  - dataloader: the DataLoader for your training data
  - device: the device (e.g., "cuda" or "cpu") on which to train the model
  - num_epochs: the number of epochs to train for
  - optimizer: your optimizer (e.g., Adam)
  - criterion: the loss function (typically nn.CrossEntropyLoss for MLM)
  - global_step: initialized to 0 before training begins
"""

best_loss = float('inf')
global_step = 0

for epoch in range(num_epochs):
    epoch_loss = 0.0
    model.train()
    # Wrap the dataloader with tqdm for a progress bar.
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)

    for batch in progress_bar:
        # Move data to device.
        # The batch contains "input_ids", "attention_mask", and "labels".
        batch = {key: value.to(device) for key, value in batch.items()}

        # Forward pass through the custom BERT model.
        # Expected output logits shape: [batch_size, sequence_length, vocab_size]
        logits = model(
            batch["input_ids"],
            batch["attention_mask"]
        )
        labels = batch["labels"]

        # Compute the MLM loss.
        # Reshape logits to [batch_size * sequence_length, vocab_size] and labels to [batch_size * sequence_length]
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Backpropagation.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        global_step += 1
        epoch_loss += loss.item()

        # Update progress bar with current loss.
        progress_bar.set_postfix(loss=loss.item())

        # Optionally, print loss every 500 training steps.
        if global_step % 500 == 0:
            print(f"Step {global_step} - Loss: {loss.item():.4f}")

    # Compute average loss for this epoch.
    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs} finished with Avg Loss: {avg_epoch_loss:.4f}")

    # Save the model if it achieves the best (lowest) epoch loss so far.
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        model_save_path = "best_trained_model.pth"  # Adjust file name/path as needed.
        torch.save(model.state_dict(), model_save_path)
        print(f"New best model saved to {model_save_path} with Avg Loss: {avg_epoch_loss:.4f}")

print("Training complete.")