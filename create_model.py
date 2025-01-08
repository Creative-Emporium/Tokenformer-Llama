import os
import math
import json
import yaml
import torch
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datasets import load_dataset
from accelerate import Accelerator
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import random_split
from typing import List, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_scheduler, PreTrainedTokenizer 

# Set up a global logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_rotary_emb(query, key, position_ids, rotary_dim):
    cos_cached = torch.cos(position_ids * (10000.0**(-torch.arange(0, rotary_dim, 2, dtype = torch.float32, device = query.device) / rotary_dim))).unsqueeze(2)
    sin_cached = torch.sin(position_ids * (10000.0**(-torch.arange(0, rotary_dim, 2, dtype = torch.float32, device = query.device) / rotary_dim))).unsqueeze(2)

    query_rot = query[..., :rotary_dim]
    key_rot = key[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    key_pass = key[..., rotary_dim:]

    query_rot_cos = query_rot * cos_cached
    query_rot_sin = rotate_half(query_rot) * sin_cached
    query_rot = query_rot_cos + query_rot_sin

    key_rot_cos = key_rot * cos_cached
    key_rot_sin = rotate_half(key_rot) * sin_cached
    key_rot = key_rot_cos + key_rot_sin
    
    query = torch.cat([query_rot, query_pass], dim = -1)
    key = torch.cat([key_rot, key_pass], dim = -1)

    return query, key

def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim = -1)
    
class GroupedQueryAttention(nn.Module):
    """
    Implements Grouped Query Attention (GQA)
    """
    def __init__(self, d_model, num_groups, dropout=0.0, use_bias_in_attn_linear=False, rotary_pct = 0.25):
        super().__init__()
        self.d_model = d_model
        self.num_groups = num_groups
        self.d_head = d_model // num_groups
        self.rotary_dim = int(d_model * rotary_pct)
        self.query_proj = nn.Linear(d_model, d_model, bias=use_bias_in_attn_linear)
        self.key_proj = nn.Linear(d_model, d_model, bias=use_bias_in_attn_linear)
        self.value_proj = nn.Linear(d_model, d_model, bias=use_bias_in_attn_linear)
        self.out_proj = nn.Linear(d_model, d_model, bias=use_bias_in_attn_linear)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, position_ids=None):
        batch_size, seq_len, _ = query.shape
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        if self.rotary_dim > 0:
          query, key = apply_rotary_emb(query, key, position_ids, self.rotary_dim)

        query = query.view(batch_size, seq_len, self.num_groups, self.d_head).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_groups, self.d_head).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_groups, self.d_head).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_head**0.5)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        return attn_output

class FeedForwardNetwork(nn.Module):
    """
    Simple Feed Forward Network
    """
    def __init__(self, d_model, expansion_factor, dropout, activation_fn="gelu", use_bias=False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion_factor, bias=use_bias)
        self.fc2 = nn.Linear(d_model * expansion_factor, d_model, bias=use_bias)
        self.dropout = nn.Dropout(dropout)
        self.activation_fn = activation_fn
        self.activation = self._get_activation(activation_fn)

    def _get_activation(self, activation_fn):
      if activation_fn == "relu":
        return F.relu
      elif activation_fn == "gelu":
        return F.gelu
      elif activation_fn == "swish":
        return F.silu # or F.swish
      elif activation_fn == "l2_norm_gelu":
        return lambda x: F.gelu(F.normalize(x, dim=-1)) # Layer norm and Gelu combination
      else:
        raise ValueError(f"Activation function {activation_fn} not supported")

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ParameterAttention(nn.Module):
    def __init__(self, d_model, num_groups, num_slots, dropout=0.0, use_bias_in_attn_linear=False):
      super().__init__()
      self.d_model = d_model
      self.num_groups = num_groups
      self.num_slots = num_slots
      self.d_head = d_model // num_groups
      self.query_proj = nn.Linear(d_model, d_model, bias = use_bias_in_attn_linear)
      self.key_proj = nn.Linear(d_model, d_model, bias = use_bias_in_attn_linear)
      self.value_proj = nn.Linear(d_model, d_model, bias = use_bias_in_attn_linear)
      self.out_proj = nn.Linear(d_model, d_model, bias = use_bias_in_attn_linear)
      self.dropout = nn.Dropout(dropout)

    def forward(self, qkv_params):
        query = self.query_proj(qkv_params)
        key = self.key_proj(qkv_params)
        value = self.value_proj(qkv_params)

        attn_scores = torch.matmul(query, key.transpose(-2,-1)) / (self.d_head**0.5)
        attn_weights = F.softmax(attn_scores, dim = -1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)
        attn_output = self.out_proj(attn_output)

        return attn_output

class TokenFormerBlock(nn.Module):
    """
    A single TokenFormer block, consisting of Parameter Attention, GQA and a Feed Forward Network.
    """
    def __init__(self, d_model, num_groups, expansion_factor, dropout, num_slots, activation_fn, use_bias_in_attn_linear, rotary_pct):
        super().__init__()
        self.parameter_attention = ParameterAttention(d_model, num_groups, num_slots, dropout, use_bias_in_attn_linear)
        self.attention = GroupedQueryAttention(d_model, num_groups, dropout, use_bias_in_attn_linear, rotary_pct)
        self.ffn = FeedForwardNetwork(d_model, expansion_factor, dropout, activation_fn, use_bias=use_bias_in_attn_linear)
        self.layer_norm1 = nn.LayerNorm(d_model, elementwise_affine = False)
        self.layer_norm2 = nn.LayerNorm(d_model, elementwise_affine = False)
        self.dropout = nn.Dropout(dropout)
        self.qkv_slots = nn.Parameter(torch.randn(3, num_slots, d_model))

    def forward(self, x, mask=None, position_ids = None):
        qkv_params = self.qkv_slots # qkv_slots are the main parameters
        qkv_params = self.parameter_attention(qkv_params)
        
        query, key, value = qkv_params[0].unsqueeze(0), qkv_params[1].unsqueeze(0), qkv_params[2].unsqueeze(0)

        attn_output = self.attention(x, key.expand(x.shape[0], -1, -1), value.expand(x.shape[0], -1, -1), mask, position_ids)
        x = self.layer_norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + self.dropout(ffn_output))
        return x

class TokenFormer(nn.Module):
    """
    Complete TokenFormer model with embedding layer and multiple TokenFormer Blocks
    """
    def __init__(self, vocab_size, d_model, n_layers, num_groups, expansion_factor, dropout, max_seq_len, num_attention_heads, qkv_slot_num, proj_slot_num, activation_fn, use_bias_in_attn_linear, rotary_pct, init_method, output_layer_init_method, norm_activation_type):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TokenFormerBlock(d_model, num_groups, expansion_factor, dropout, qkv_slot_num, activation_fn, use_bias_in_attn_linear, rotary_pct) for _ in range(n_layers)
        ])
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.num_attention_heads = num_attention_heads
        self.qkv_slot_num = qkv_slot_num
        self.proj_slot_num = proj_slot_num
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine = True)
        self.norm_activation_type = norm_activation_type
        self.init_method = init_method
        self.output_layer_init_method = output_layer_init_method
        self.apply(self._init_weights)
        self.apply(self._init_output_layer)

    def _init_weights(self, module):
        """
        Initializes the weights of the model.
        Args:
            module (nn.Module): model module
        """
        if isinstance(module, nn.Linear):
            if self.init_method == "normal":
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            elif self.init_method == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                 torch.nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if self.init_method == "normal":
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            elif self.init_method == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(module.weight)
            else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)

    def _init_output_layer(self, module):
        if isinstance(module, nn.Linear) and module == self.output_projection:
              if self.output_layer_init_method == "normal":
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
              elif self.output_layer_init_method == "wang_init":
                    torch.nn.init.normal_(module.weight, mean=0.0, std=(2.0/math.sqrt(self.d_model*2)))
              else:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
              if module.bias is not None:
                  torch.nn.init.zeros_(module.bias)
                  
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(x)
        x = x + self.position_embedding(positions)
        for layer in self.layers:
            x = layer(x, mask, positions)
        x = self.final_norm(x)
        logits = self.output_projection(x)
        return logits

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_json_config(config, path="config.json"):
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_json_config(path="config.json"):
     with open(path, 'r') as f:
          return json.load(f)

class TextDatasetProcessor:
    @staticmethod
    def remove_blank_lines(text: str) -> str:
        return "\n".join([line for line in text.splitlines() if line.strip()])

class InferenceEngine:
    def __init__(self, model, tokenizer, max_new_tokens: int = 50):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def generate_text(self, prompt: Optional[str] = None, open_ended: bool = False, **gen_kwargs: Dict[str, Any]) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids if prompt else torch.tensor([[self.tokenizer.pad_token_id]])
        output = self.model.generate(input_ids=input_ids, max_new_tokens=self.max_new_tokens, **gen_kwargs)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

class TrainingLogger:
    def __init__(self):
        self.total_loss = 0.0
        self.total_val_loss = 0.0
        self.total_grad_norm = 0.0
        self.total_tokens = 0
        self.steps = 0
        self.val_steps = 0
        self.train_losses = []
        self.val_losses = []
        self.train_perplexity = []
        self.val_perplexity = []

    def log_step(self, loss: float, grad_norm: float, tokens: int, is_train_step: bool = True):
        if is_train_step:
            self.total_loss += loss
            self.total_grad_norm += grad_norm
            self.total_tokens += tokens
            self.steps += 1
            self.train_losses.append(loss)
            self.train_perplexity.append(math.exp(loss))
            logger.info(f"Train Step {self.steps}: Loss = {loss:.4f}, Gradient Norm = {grad_norm:.4f}, Tokens Seen = {self.total_tokens}")
        else:
            self.total_val_loss += loss
            self.val_steps +=1
            self.val_losses.append(loss)
            self.val_perplexity.append(math.exp(loss))
            logger.info(f"Validation Step {self.val_steps}: Validation Loss = {loss:.4f}")

    def log_final_summary(self):
        avg_loss = self.total_loss / self.steps if self.steps else 0
        avg_val_loss = self.total_val_loss / self.val_steps if self.val_steps else 0
        avg_grad_norm = self.total_grad_norm / self.steps if self.steps else 0
        logger.info(f"Training Complete: Average Training Loss = {avg_loss:.4f}, "
                    f"Average Validation Loss = {avg_val_loss:.4f}, "
                    f"Average Gradient Norm = {avg_grad_norm:.4f}, "
                    f"Total Tokens Seen = {self.total_tokens}")

    def plot_losses_and_perplexity(self):
      epochs = range(1, len(self.train_losses) // len(self.train_losses)+1)
      fig, axs = plt.subplots(2, 1, figsize = (10,8))

      axs[0].plot(epochs, self.train_losses, label = "Training Loss")
      if self.val_losses:
        axs[0].plot(epochs, self.val_losses, label = "Validation Loss")
      axs[0].set_xlabel("Epochs")
      axs[0].set_ylabel("Loss")
      axs[0].set_title("Training and Validation Loss")
      axs[0].legend()
      axs[0].grid(True)

      axs[1].plot(epochs, self.train_perplexity, label = "Training Perplexity")
      if self.val_perplexity:
          axs[1].plot(epochs, self.val_perplexity, label = "Validation Perplexity")
      axs[1].set_xlabel("Epochs")
      axs[1].set_ylabel("Perplexity")
      axs[1].set_title("Training and Validation Perplexity")
      axs[1].legend()
      axs[1].grid(True)

      plt.tight_layout()
      plt.savefig('loss_and_perplexity_plot.png')
      plt.show()

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_seq_len):
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
        texts = [str(x['text']) for x in dataset]
    else:
        raise ValueError("Please specify a valid dataset option")

    dataset = TextDataset(texts, tokenizer, max_seq_len)
    return dataset

def scale_model_params(model, scale_factor):
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
              new_embedding_size = int(module.weight.shape[1] * scale_factor)
              with torch.no_grad():
                original_weight = module.weight.clone().detach()
                new_weight = torch.randn(module.num_embeddings, new_embedding_size, device = module.weight.device, dtype = module.weight.dtype)
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

def get_num_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params/1000000

def create_initial_config(config, tokenizer):
    """
    Creates the initial config.json file independently.
    Args:
      config (dict): config dictionary
      tokenizer (transformers.PreTrainedTokenizer): The tokenizer object
    Returns:
        config (dict): json config dict
    """
    config_json = {}
    config_json['model_type'] = "tokenformer"
    config_json['architectures'] = ["TokenFormerForCausalLM"]
    config_json['tie_word_embeddings'] = False
    config_json['initializer_range'] = 0.02
    config_json['use_cache'] = True
    config_json['position_embedding_type'] = 'rotary'
    config_json['attention_bias'] = False
    config_json['torch_dtype'] = "float32"
    config_json['layer_norm_eps'] = 1e-5 # default value
    config_json['rope_theta'] = 10000.0 # default value
    config_json['use_flash_attention'] = config.get("use_flash_attention", False)

    config_json['bos_token_id'] = tokenizer.bos_token_id
    config_json['eos_token_id'] = tokenizer.eos_token_id
    config_json['pad_token_id'] = tokenizer.pad_token_id

    config_json['d_model'] = config['d_model']
    config_json['n_layers'] = config['n_layers']
    config_json['num_groups'] = config['gqa_num_groups']
    config_json['intermediate_size'] = config['d_model'] * config['expansion_factor']
    config_json['hidden_act'] = config.get('activation_fn', "relu")
    config_json['max_position_embeddings'] = config['max_seq_len']
    config_json['model_max_length'] = config['max_seq_len']
    config_json['vocab_size'] = len(tokenizer)
    config_json['dropout'] = config['dropout']
    config_json['qkv_slot_num'] = config['d_model']
    config_json['proj_slot_num'] = config['d_model']
    config_json['ffn_slot_num'] = config['d_model'] * config['expansion_factor']
    config_json['num_attention_heads'] = config['gqa_num_groups']
    config_json['use_bias_in_attn_linear'] = config.get("use_bias_in_attn_linear", False)
    config_json['rotary_pct'] = config.get("rotary_pct", 0.25)
    config_json['norm_activation_type'] = config.get('norm_activation_type', "gelu")
    config_json['init_method'] = config.get('init_method', 'normal')
    config_json['output_layer_init_method'] = config.get('output_layer_init_method', 'wang_init')
    
    return config_json

def scale_init_model(model, config):
    scale_factor = math.sqrt(config['init_model_size'] / get_num_params(model))
    model = scale_model_params(model, scale_factor)
    return model

class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.accelerator = Accelerator(gradient_accumulation_steps=config['gradient_accumulation_steps'])
        self.train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        self.val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False) if val_dataset else None
        self.optimizer = self._get_optimizer(model, config)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer, config)
        self.criterion = nn.CrossEntropyLoss()
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(
            model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler
        )
        self.config = config
        self.training_logger = TrainingLogger()
        self.tokenizer = AutoTokenizer

    def _get_optimizer(self, model, config):
      optimizer_type = config.get('optimizer', "adamw").lower()
      learning_rate = config.get('learning_rate', 1e-4)
      weight_decay = config.get('weight_decay', 1e-5)

      if optimizer_type == "adamw":
          return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
      elif optimizer_type == "adan":
          return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
      elif optimizer_type == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        except ImportError:
          raise ImportError("Lion Optimizer not found. Please install lion-pytorch with `pip install lion-pytorch`")
      else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    def _get_lr_scheduler(self, optimizer, config):
      scheduler_type = config.get('lr_scheduler', "linear").lower()
      num_epochs = config.get('num_epochs', 10)
      warmup_steps = config.get('warmup_steps', 0)
      total_training_steps = len(self.train_dataloader) * num_epochs

      if scheduler_type == "linear":
          return get_scheduler(
              name='linear',
              optimizer=optimizer,
              num_warmup_steps = warmup_steps,
              num_training_steps = total_training_steps,
              )
      elif scheduler_type == "cosine":
        return get_scheduler(
              name = 'cosine',
              optimizer = optimizer,
              num_warmup_steps=warmup_steps,
              num_training_steps = total_training_steps,
        )
      else:
        return get_scheduler(
            name = 'linear',
            optimizer = optimizer,
            num_warmup_steps = warmup_steps,
            num_training_steps = total_training_steps
        )

    def _compute_loss(self, outputs, input_ids):
        return self.criterion(outputs.view(-1, outputs.shape[-1]), input_ids.view(-1))

    def _compute_gradient_norm(self):
      total_norm = 0
      for p in self.accelerator.unwrap_model(self.model).parameters():
        if p.grad is not None:
           param_norm = p.grad.detach().data.norm(2)
           total_norm += param_norm.item()**2
      total_norm = total_norm**0.5
      return total_norm

    def _clip_gradients(self):
        grad_clip = self.config.get('gradient_clipping', 1.0)
        if grad_clip:
           clip_grad_norm_(self.accelerator.unwrap_model(self.model).parameters(), grad_clip)

    def _train_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        outputs = self.model(input_ids, mask=attention_mask)
        loss = self._compute_loss(outputs, input_ids)
        self.accelerator.backward(loss)
        self._clip_gradients()
        grad_norm = self._compute_gradient_norm()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()
        num_tokens = torch.numel(input_ids)
        # Convert tensors to floats before returning
        return loss.detach().item(), grad_norm, float(num_tokens)

    def _validation_step(self, batch):
       input_ids = batch['input_ids']
       attention_mask = batch['attention_mask']
       outputs = self.model(input_ids, mask=attention_mask)
       loss = self._compute_loss(outputs, input_ids)
       return loss.detach()# returns loss as a detached tensor


    def train(self, save_path="trained_model"):
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} (Training)")

            for batch in progress_bar:
                loss, grad_norm, num_tokens = self._train_step(batch)
                # Gather metrics from all processes
                gathered_loss, gathered_grad_norm, gathered_num_tokens = self.accelerator.gather_for_metrics((loss, grad_norm, num_tokens))
                
                # Since the gathered values are already floats, we don't need to call .item()
                self.training_logger.log_step(gathered_loss, gathered_grad_norm, gathered_num_tokens)
                progress_bar.set_postfix(loss=gathered_loss)

            if self.val_dataloader:
                self.model.eval()
                val_progress_bar = tqdm(self.val_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']} (Validation)")
                with torch.no_grad():
                    for batch in val_progress_bar:
                        loss = self._validation_step(batch)
                        gathered_loss = self.accelerator.gather_for_metrics(loss)
                        # Since the gathered loss is already a float, we don't need to call .item()
                        self.training_logger.log_step(gathered_loss, grad_norm=0, tokens=0, is_train_step=False)
                        val_progress_bar.set_postfix(val_loss=gathered_loss)

        self.training_logger.log_final_summary()
        self.training_logger.plot_losses_and_perplexity()        
        self._save_model(save_path)

    def _save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
           torch.save(self.accelerator.unwrap_model(self.model).state_dict(), os.path.join(save_path, "model.pth"))
           self.tokenizer.save_pretrained(save_path)
           save_json_config(self.config, os.path.join(save_path, "config.json"))
           logger.info(f"Model, tokenizer and configs saved at: {save_path}")
        self.accelerator.end_training()

    def load_model_from_pretrained(self, model_path):
         """
            Loads a model from the given path.
            Args:
              model_path (str): path to model
              config (dict): Dictionary of configuration params
            Returns:
              model (nn.Module): Returns model with loaded params
         """
         state_dict = torch.load(os.path.join(model_path, 'model.pth'), map_location=torch.device('cpu'))
         self.accelerator.unwrap_model(self.model).load_state_dict(state_dict)
         logger.info(f"Model loaded from {model_path}")
         return self.model

def main():
    # Parse args
    parser = argparse.ArgumentParser(description="Tokenformer GQA training script")
    parser.add_argument("--config_path", type=str, default='config.yaml', help="Path to the config file")
    parser.add_argument("--load_model_path", type = str, default = None, help="Path to pretrained model")
    args = parser.parse_args()

    # Load the configurations
    config = load_config(args.config_path)

    # Load dataset
    full_dataset = load_text_data(config)
    tokenizer = full_dataset.tokenizer

    # Create train/val split
    train_size = int(config['train_ratio'] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size]) if val_size > 0 else (full_dataset, None)

    # Create Initial Config.json
    initial_json_config = create_initial_config(config, tokenizer)

    # Initialize the model
    model = TokenFormer(
        vocab_size=len(tokenizer),
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        num_groups=config['gqa_num_groups'],
        expansion_factor=config['expansion_factor'],
        dropout=config['dropout'],
        max_seq_len=config['max_seq_len'],
        num_attention_heads=config['gqa_num_groups'],
        qkv_slot_num=config['d_model'],
        proj_slot_num=config['d_model'],
        activation_fn = config.get('activation_fn', 'relu')
    )
    
    # Load Model params
    if args.load_model_path:
         trainer = ModelTrainer(model, train_dataset, val_dataset, config)
         model = trainer.load_model_from_pretrained(args.load_model_path)
         print(f"Model size: {get_num_params(model)} million params") # only prints model size if model is loaded from a saved file
    else:
      if config['scale_model']:
        # Scaled initialization
        model = scale_init_model(model, config)
        # Scale the model size if flag is set
        model = scale_model_params(model, config['scale_factor'])
        print(f"Model size: {get_num_params(model)} million params")
      else:
        print(f"Initial Model size: {get_num_params(model)} million params")

      trainer = ModelTrainer(model, train_dataset, val_dataset, config) # initializes training
      trainer.train() # train the model

    # Update and save config to have updated model size
    initial_json_config['num_params'] = get_num_params(model)
    save_json_config(initial_json_config, os.path.join('trained_model',"config.json"))
    print(f"Updated config with current model size and saved at trained_model/config.json")

if __name__ == '__main__':
    main()
