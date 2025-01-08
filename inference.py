import os
import torch
import argparse
import yaml
from transformers import AutoTokenizer
from torch.nn import functional as F
from accelerate import Accelerator

# Model classes (same as in training code)
import math
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA) as described in:
    https://arxiv.org/pdf/2305.13245.pdf
    """
    def __init__(self, d_model, num_groups, dropout=0.0):
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
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion_factor)
        self.fc2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TokenFormerBlock(nn.Module):
    """
    A single TokenFormer block, consisting of GQA and a Feed Forward Network
    """
    def __init__(self, d_model, num_groups, expansion_factor, dropout):
        super().__init__()
        self.attention = GroupedQueryAttention(d_model, num_groups, dropout)
        self.ffn = FeedForwardNetwork(d_model, expansion_factor, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
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

def load_config(path):
  """
  Loads a config file
  Args:
     path (str): Path to config
  Returns:
     config (dict): Config dict
  """
  with open(path, 'r') as f:
    return yaml.safe_load(f)

def load_model_and_tokenizer(model_path):
    """
    Loads the trained model and tokenizer.
    Args:
      model_path (str): Path to the saved model directory.
    Returns:
      tuple: model and tokenizer objects
    """
    config_path = os.path.join(model_path, 'config.yaml')
    config = load_config(config_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_size = len(tokenizer)
    model = TokenFormer(
      vocab_size=vocab_size,
      d_model=config['d_model'],
      n_layers=config['n_layers'],
      num_groups=config['gqa_num_groups'],
      expansion_factor=config['expansion_factor'],
      dropout=config['dropout'],
      max_seq_len=config['max_seq_len']
    )
    model_state_dict_path = os.path.join(model_path, 'model.pth')
    model.load_state_dict(torch.load(model_state_dict_path, map_location = "cpu"))
    return model, tokenizer, config


def generate_text(model, tokenizer, config, prompt, max_length=50, temperature=1.0, top_k=10):
    """
    Generates text using the trained model.
    Args:
        model (nn.Module): Trained TokenFormer model.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer instance.
        config (dict): Configuration dictionary.
        prompt (str): Initial text prompt.
        max_length (int, optional): Maximum length of generated text. Defaults to 50.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        top_k (int, optional): Top-k sampling value. Defaults to 10.
    Returns:
       str: Generated text.
    """
    model.eval()
    accelerator = Accelerator()
    model = accelerator.prepare(model)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(accelerator.device)
    generated_ids = input_ids

    with torch.no_grad():
      for _ in range(max_length):
            outputs = model(generated_ids) # Pass generated_ids as input
            next_token_logits = outputs[:, -1, :]  # Get the logits for the last token
            
            # Temperature scaling
            scaled_logits = next_token_logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(scaled_logits, top_k, dim=-1)
                min_value = values[:, -1].unsqueeze(-1)
                scaled_logits = torch.where(scaled_logits < min_value, torch.tensor(-1e10, device=scaled_logits.device), scaled_logits)

            probs = F.softmax(scaled_logits, dim=-1) # Get probabilities from scaled logits
            next_token = torch.multinomial(probs, num_samples=1) # Sample next token

            generated_ids = torch.cat((generated_ids, next_token), dim=1) # Concatenate generated_ids and new generated token

            if next_token == tokenizer.eos_token_id:
                break


    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text


def main():
    """
    Main function to perform text generation.
    """
    parser = argparse.ArgumentParser(description="Text generation script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial text prompt")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the generated text")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling")
    args = parser.parse_args()

    model, tokenizer, config = load_model_and_tokenizer(args.model_path)
    generated_text = generate_text(model, tokenizer, config, args.prompt, args.max_length, args.temperature, args.top_k)
    print(f"Generated Text:\n{generated_text}")


if __name__ == "__main__":
    main()
