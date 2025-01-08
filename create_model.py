import os
import math
import yaml
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datasets import load_dataset
from accelerate import Accelerator
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import json

class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) as described in:
    https://arxiv.org/pdf/2305.13245.pdf
    """
    def __init__(self, d_model, num_groups, dropout=0.0):
        """
        Initializes the GQA module.
        Args:
            d_model (int): The model's embedding dimension.
            num_groups (int): The number of query groups.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """      
        super().__init__()
        self.d_model = d_model
        self.num_groups = num_groups
        self.d_head = d_model // num_groups  # dimension of each head

        # Query, Key, Value projection matrices
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Implements grouped query attention
        Args:
            query (torch.Tensor): Input query tensor (batch_size, seq_len, d_model)
            key (torch.Tensor): Input key tensor (batch_size, seq_len, d_model)
            value (torch.Tensor): Input value tensor (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask (batch_size, seq_len)
        Returns:
            torch.Tensor: Attention output (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = query.shape

        # Linear projections
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # Reshape for GQA
        query = query.view(batch_size, seq_len, self.num_groups, self.d_head).transpose(1, 2)  # (batch_size, num_groups, seq_len, d_head)
        key = key.view(batch_size, seq_len, self.num_groups, self.d_head).transpose(1, 2)      # (batch_size, num_groups, seq_len, d_head)
        value = value.view(batch_size, seq_len, self.num_groups, self.d_head).transpose(1, 2)  # (batch_size, num_groups, seq_len, d_head)

        # Attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_head**0.5)  # (batch_size, num_groups, seq_len, seq_len)

        # Mask application
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1) # (batch_size, 1, 1, seq_len)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_groups, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, value)  # (batch_size, num_groups, seq_len, d_head)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)  # (batch_size, seq_len, d_model)

        # Output projection
        attn_output = self.out_proj(attn_output)
        return attn_output

class FeedForwardNetwork(nn.Module):
    """
    Simple Feed Forward Network
    """
    def __init__(self, d_model, expansion_factor, dropout):
        """
        Initializes the FFN module.
        Args:
            d_model (int): The model's embedding dimension.
            expansion_factor (int): The expansion factor for the FFN hidden layer.
            dropout (float): Dropout probability.
        """        
      super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion_factor)
        self.fc2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Implements the forward pass of the FFN.
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).
        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, d_model).
        """      
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TokenFormerBlock(nn.Module):
    """
    A single TokenFormer block, consisting of GQA and a Feed Forward Network.
    """
    def __init__(self, d_model, num_groups, expansion_factor, dropout):
        """
        Initializes a TokenFormer block.
        Args:
            d_model (int): The model's embedding dimension.
            num_groups (int): The number of query groups in GQA.
            expansion_factor (int): Expansion factor for the FFN.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.attention = GroupedQueryAttention(d_model, num_groups, dropout)
        self.ffn = FeedForwardNetwork(d_model, expansion_factor, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Implements the forward pass of the TokenFormer block.
        Args:
            x (torch.Tensor): Input tensor (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask (batch_size, seq_len).
        Returns:
            torch.Tensor: Output tensor (batch_size, seq_len, d_model).
        """        
        # GQA
        attn_output = self.attention(x, x, x, mask)
        # Residual connection and LayerNorm
        x = self.layer_norm1(x + self.dropout(attn_output))
        # FFN
        ffn_output = self.ffn(x)
        # Residual connection and LayerNorm
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x

class TokenFormer(nn.Module):
    """
    Complete TokenFormer model with embedding layer and multiple TokenFormer Blocks
    """
    def __init__(self, vocab_size, d_model, n_layers, num_groups, expansion_factor, dropout, max_seq_len):
        """
        Initializes the TokenFormer model.
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): The model's embedding dimension.
            n_layers (int): Number of TokenFormer blocks.
            num_groups (int): Number of query groups in GQA.
            expansion_factor (int): Expansion factor for FFN.
            dropout (float): Dropout probability.
            max_seq_len (int): Maximum sequence length for positional embeddings.
        """        
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)  # for positional embedding
        self.layers = nn.ModuleList([
            TokenFormerBlock(d_model, num_groups, expansion_factor, dropout) for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        Forward pass through the TokenFormer model
        Args:
            x (torch.Tensor): Input tensor of token ids (batch_size, seq_len)
            mask (torch.Tensor, optional): Attention mask (batch_size, seq_len)
        Returns:
            torch.Tensor: Output logits (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        x = x + self.position_embedding(positions)  # positional embeddings

        for layer in self.layers:
            x = layer(x, mask)
        logits = self.output_projection(x)
        return logits

def load_config(path="config.yaml"):
    """
    Loads configurations from a YAML file.
    Args:
      path (str): Configuration file
    Returns:
        dict: dictionary representing the config
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config, path="config.yaml"):
    """
    Saves configurations to a YAML file.
    Args:
      config (dict): Config dictionary
      path (str): Path to config file
    Returns:
      None
    """
    with open(path, 'w') as f:
        yaml.dump(config, f, indent=4)

class TextDataset(Dataset):
    """
    Custom Dataset class for handling text data.
    """
    def __init__(self, texts, tokenizer, max_seq_len):
        """
        Initializes the TextDataset.
        Args:
            texts (list): List of text strings.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
            max_seq_len (int): Maximum sequence length.
        """      
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokenized = self.tokenizer(text,
                                  max_length=self.max_seq_len,
                                  padding='max_length',
                                  truncation=True,
                                  return_tensors='pt')

        return {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0)
        }

def load_text_data(config):
    """
    Loads and prepares text data either from a local file or an online source
    Args:
        config (dict): Configuration dictionary
    Returns:
        TextDataset: Dataset instance
    """
    tokenizer = AutoTokenizer.from_pretrained("artificialguybr/LLAMA-3.2-1B-OpenHermes2.5")
    local_path = config.get('data_path')
    online_dataset = config.get('online_dataset')
    dataset_name = config.get('dataset_name')
    dataset_config = config.get('dataset_config')
    max_seq_len = config.get('max_seq_len')
    local_dataset = config.get('local_dataset')

    if local_dataset:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local dataset not found at: {local_path}")
        with open(local_path, 'r', encoding='utf-8') as file:
            texts = file.readlines()
    elif online_dataset:
        if not dataset_name or not dataset_config:
            raise ValueError("Please provide a valid huggingface dataset name and configuration")
        dataset = load_dataset(dataset_name, dataset_config, split="train")
        texts = [str(x['text']) for x in dataset]  # Assumes "text" is the key
    else:
        raise ValueError("Please specify a valid dataset option")

    dataset = TextDataset(texts, tokenizer, max_seq_len)
    return dataset


def scale_model_params(model, scale_factor):
    """
    Scales model by increasing embedding/hidden sizes via linear scaling
    Args:
      model (nn.Module): Initial model
      scale_factor (float): Factor by which to scale model
    Returns:
      model (nn.Module): Scaled model
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            if hasattr(module, "in_features") and module.in_features != 0:
              module.in_features = int(module.in_features * scale_factor)
            if hasattr(module, "out_features")and module.out_features != 0:
              module.out_features = int(module.out_features * scale_factor)
            if hasattr(module, "num_embeddings") and module.num_embeddings !=0 :
              module.num_embeddings = int(module.num_embeddings * scale_factor)
            if hasattr(module, "normalized_shape") and isinstance(module.normalized_shape, tuple):
              new_shape = (int(dim * scale_factor) for dim in module.normalized_shape)
              module.normalized_shape = tuple(new_shape)

        if isinstance(module, nn.Embedding):
            if hasattr(module, "embedding_dim"):
              module.embedding_dim = int(module.embedding_dim * scale_factor)
            if hasattr(module, "weight"):
              new_embedding_size = int(module.weight.shape[1] * scale_factor)  # scales embedding dim
              with torch.no_grad():
                original_weight = module.weight.clone().detach()
                new_weight = torch.randn(module.num_embeddings, new_embedding_size, device = module.weight.device, dtype = module.weight.dtype) # creates new random params
                new_weight[:,:module.weight.shape[1]] = original_weight
                module.weight = torch.nn.Parameter(new_weight)


    for name, module in model.named_modules():
      if isinstance(module, nn.Linear):
        with torch.no_grad():
          if module.weight.shape[0] != 0 and module.weight.shape[1] != 0:
            original_weight = module.weight.clone().detach()
            module.weight = torch.nn.Parameter(torch.randn(module.out_features, module.in_features, device = module.weight.device, dtype = module.weight.dtype))
            module.weight[:, :original_weight.shape[1]] = original_weight
      if isinstance(module, nn.LayerNorm):
        with torch.no_grad():
          if module.weight is not None and module.bias is not None and len(module.weight) !=0 and len(module.bias) != 0:
            original_weight = module.weight.clone().detach()
            original_bias = module.bias.clone().detach()
            module.weight = torch.nn.Parameter(torch.randn(len(module.normalized_shape), device = module.weight.device, dtype = module.weight.dtype))
            module.bias = torch.nn.Parameter(torch.zeros(len(module.normalized_shape), device = module.bias.device, dtype = module.bias.dtype))
            module.weight[:len(original_weight)] = original_weight
            module.bias[:len(original_bias)] = original_bias
    return model

def train_model(model, train_dataset, config, save_path="trained_model"):
    """
    Trains the TokenFormer model using the given dataset.
    Args:
        model (nn.Module): TokenFormer model
        train_dataset (Dataset): training dataset
        config (dict): config dictionary
        save_path (str): Path to save trained model
    """
    accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    for epoch in range(config['num_epochs']):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        total_loss = 0
        for batch in progress_bar:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            outputs = model(input_ids, mask=attention_mask) # Use attention mask
            loss = criterion(outputs.view(-1, outputs.shape[-1]), input_ids.view(-1))
            total_loss += accelerator.gather(loss).item()

            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.set_postfix(loss=total_loss / (progress_bar.n + 1))
    print(f"Model trained successfully.")

    # Save trained model
    os.makedirs(save_path, exist_ok=True)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), os.path.join(save_path, "model.pth"))
        train_dataset.tokenizer.save_pretrained(save_path)
        save_config(config, os.path.join(save_path, "config.yaml"))  # save the config in the same dir
        print(f"Model and tokenizer saved at: {save_path}")
    accelerator.end_training()

def scale_init_model(model, config):
    """
      Scales the model based on size
      Args:
        model (nn.Module): Initial model
        config (dict): Configuration dictionary
      Returns:
        model (nn.Module): Scaled model
    """
    scale_factor = math.sqrt(config['init_model_size'] / get_num_params(model))  # Scale factor for model scaling
    model = scale_model_params(model, scale_factor) # Scales the model up to size
    return model

def get_num_params(model):
   """
   Gets the number of parameters
   Args:
     model (nn.Module): Model
   Returns:
     int : number of params
   """
   total_params = sum(p.numel() for p in model.parameters())  # total number of trainable params in model
   return total_params/1000000 # params in millions


def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Tokenformer GQA training script")
    parser.add_argument("--config_path", type=str, default='config.yaml', help="Path to the config file")
    args = parser.parse_args()

    # Load the configurations
    config = load_config(args.config_path)

    # Load dataset
    train_dataset = load_text_data(config)
    vocab_size = len(train_dataset.tokenizer)

    # Initialize the model
    model = TokenFormer(vocab_size=vocab_size,
                        d_model=config['d_model'],
                        n_layers=config['n_layers'],
                        num_groups=config['gqa_num_groups'],
                        expansion_factor=config['expansion_factor'],
                        dropout=config['dropout'],
                        max_seq_len=config['max_seq_len'])
    if config['scale_model']:
        # Scaled initialization
        model = scale_init_model(model, config)
        # Scale the model size if flag is set
        model = scale_model_params(model, config['scale_factor'])
        print(f"Model size: {get_num_params(model)} million params")
    else:
        print(f"Initial Model size: {get_num_params(model)} million params") #print initial model size if not scaling on initialization
    train_model(model, train_dataset, config)


    # Update and save config to have updated model size
    config['num_params'] = get_num_params(model)
    save_config(config, args.config_path) # Save updated config file

    print(f"Updated config with current model size and saved at {args.config_path}")

if __name__ == '__main__':
    main()
