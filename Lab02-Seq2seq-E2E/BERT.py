import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define your custom BERT model class with reduced parameters.
class BertModel(nn.Module):
    def __init__(
            self,
            vocab_size=30522,  # We use the vocab size from the pretrained tokenizer
            hidden_size=768,
            max_position_embeddings=128,  # Reduced maximum sequence length
            num_attention_heads=4,  # Reduced number of attention heads
            num_hidden_layers=4,  # Reduced number of transformer layers
            intermediate_size=3072,
            dropout=0.1):
        """
        A simplified BERT-like model using PyTorch's TransformerEncoder.
        """
        super(BertModel, self).__init__()

        # Embedding layers for token, position, and token type embeddings:
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)  # typically for sentence A and B

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

        # Create a stack of transformer encoder layers.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation='gelu'  # BERT uses the GELU non-linearity
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)

    def forward(self, input_ids, token_type_ids=None):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            token_type_ids: (Optional) Tensor of shape (batch_size, seq_length)
        Returns:
            encoder_output: Tensor of shape (batch_size, seq_length, hidden_size)
        """
        device = input_ids.device
        batch_size, seq_length = input_ids.size()

        # Create position ids: [0, 1, 2, ... , seq_length-1] for every example in the batch
        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, seq_length)

        # If token_type_ids are not provided, create a zero tensor
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Sum the embeddings for tokens, positions, and token types
        word_embeds = self.word_embeddings(input_ids)
        pos_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)

        embeddings = word_embeds + pos_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # TransformerEncoder expects input in shape: (seq_length, batch_size, hidden_size)
        embeddings = embeddings.transpose(0, 1)
        encoder_output = self.encoder(embeddings)
        encoder_output = encoder_output.transpose(0, 1)

        return encoder_output


class BertForMaskedLM(nn.Module):
    def __init__(self, config):
        """
        Wrap the BertModel with a head on top for masked language modeling.
        """
        super(BertForMaskedLM, self).__init__()
        self.bert = BertModel(
            vocab_size=config['vocab_size'],
            hidden_size=config['hidden_size'],
            max_position_embeddings=config['max_position_embeddings'],
            num_attention_heads=config['num_attention_heads'],
            num_hidden_layers=config['num_hidden_layers'],
            intermediate_size=config['intermediate_size'],
            dropout=config['dropout']
        )
        # The output head is simply a linear layer projecting the hidden states
        # to the vocabulary size. We tie the output weights with the word_embeddings.
        self.lm_head = nn.Linear(config['hidden_size'], config['vocab_size'], bias=False)

        # Weight tying: set lm_head weights equal to the word embeddings.
        self.lm_head.weight = self.bert.word_embeddings.weight

        # Define loss function; ignore index -100 (commonly used to ignore labels)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, token_type_ids=None, labels=None):
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            token_type_ids: (Optional) Tensor of shape (batch_size, seq_length)
            labels: (Optional) Tensor of shape (batch_size, seq_length). Masked positions have
                    the correct token ids and non-masked positions are set to -100.
        Returns:
            If labels is provided, returns (loss, prediction_scores)
            else, returns prediction_scores.
        """
        sequence_output = self.bert(input_ids, token_type_ids)
        # Compute scores over vocabulary for each token in the sequence.
        prediction_scores = self.lm_head(sequence_output)

        if labels is not None:
            # Reshape for loss computation: (batch_size * seq_length, vocab_size)
            loss = self.loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1)
            )
            return loss, prediction_scores
        else:
            return prediction_scores