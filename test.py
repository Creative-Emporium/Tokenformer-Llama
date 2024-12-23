import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from typing import Optional, List, Union, Dict
import wandb
import json
import os
from transformers import AutoTokenizer
import logging
from pathlib import Path

class TrainingConfig:
    def __init__(
        self,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        batch_size: int = 32,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        num_epochs: Optional[int] = None,
        output_dir: str = "checkpoints",
        use_wandb: bool = False
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.num_epochs = num_epochs
        self.output_dir = output_dir
        self.use_wandb = use_wandb


def evaluate_model(model, eval_dataloader, criterion, device):
    """Evaluate the model on the evaluation dataset."""
    model.eval()
    total_eval_loss = 0
    
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            shift_logits = outputs[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_eval_loss += loss.item()
    
    model.train()
    return total_eval_loss / len(eval_dataloader)

def generate_sample(model, tokenizer, device, prompt="The robot said:", max_length=50):
    """Generate a sample text using the current model state."""
    return inference(model, tokenizer, prompt, device=device, max_length=max_length)

class TextDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        file_path: Optional[str] = None,
        dataset_name: Optional[str] = None,
        max_length: int = 1024,
        text_column: str = "text",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the dataset from either a local file or HuggingFace dataset.
        
        Args:
            tokenizer: Tokenizer to use for encoding text
            file_path: Path to local text file (optional)
            dataset_name: Name of HuggingFace dataset (optional)
            max_length: Maximum sequence length
            text_column: Column name containing text for HuggingFace datasets
            cache_dir: Directory to cache processed datasets
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        if file_path and dataset_name:
            raise ValueError("Specify either file_path or dataset_name, not both")
        elif file_path:
            self.examples = self._load_local_file(file_path)
        elif dataset_name:
            self.examples = self._load_huggingface_dataset(dataset_name, text_column)
        else:
            raise ValueError("Must specify either file_path or dataset_name")
            
        # Create cache directory if it doesn't exist
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_path = Path(cache_dir) / f"processed_dataset_{hash(str(file_path or dataset_name))}.pt"
        else:
            self.cache_path = None

    def _load_local_file(self, file_path: str) -> List[str]:
        """Load and preprocess text from a local file."""
        logging.info(f"Loading text from {file_path}")
        
        # Handle different file formats
        ext = Path(file_path).suffix
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext == '.jsonl':
            import jsonlines
            texts = []
            with jsonlines.open(file_path) as reader:
                for obj in reader:
                    if isinstance(obj, dict):
                        texts.append(obj.get('text', ''))
                    else:
                        texts.append(str(obj))
            text = '\n'.join(texts)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Split into chunks of approximately max_length tokens
        tokens = self.tokenizer.encode(text)
        chunks = [tokens[i:i + self.max_length] for i in range(0, len(tokens), self.max_length)]
        
        # Convert back to text for storage
        return [self.tokenizer.decode(chunk) for chunk in chunks]

    def _load_huggingface_dataset(self, dataset_name: str, text_column: str) -> List[str]:
        """Load and preprocess text from a HuggingFace dataset."""
        logging.info(f"Loading dataset {dataset_name}")
        
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name)
        
        # Get the first split (usually 'train')
        split = list(dataset.keys())[0]
        texts = dataset[split][text_column]
        
        # Concatenate texts and split into chunks
        full_text = ' '.join(texts)
        tokens = self.tokenizer.encode(full_text)
        chunks = [tokens[i:i + self.max_length] for i in range(0, len(tokens), self.max_length)]
        
        return [self.tokenizer.decode(chunk) for chunk in chunks]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        
        # Tokenize with padding and truncation
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Remove batch dimension added by tokenizer
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0)
        }

def create_dataloaders(
    tokenizer,
    batch_size: int = 32,
    max_length: int = 1024,
    num_workers: int = 4,
    file_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    train_test_split: float = 0.1,
    seed: int = 42,
    cache_dir: Optional[str] = None
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        tokenizer: Tokenizer to use for encoding text
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        num_workers: Number of workers for DataLoader
        file_path: Path to local text file (optional)
        dataset_name: Name of HuggingFace dataset (optional)
        train_test_split: Fraction of data to use for validation
        seed: Random seed for reproducibility
        cache_dir: Directory to cache processed datasets
    
    Returns:
        Tuple of (train_dataloader, eval_dataloader)
    """
    # Create dataset
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        dataset_name=dataset_name,
        max_length=max_length,
        cache_dir=cache_dir
    )
    
    # Split into train and validation sets
    total_size = len(dataset)
    val_size = int(train_test_split * total_size)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    ) if val_size > 0 else None
    
    return train_dataloader, eval_dataloader

# --- Helper Classes (RMSNorm, RotaryEmbedding, etc.) ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm_inv = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm_inv * self.weight

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1. / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return torch.sin(emb), torch.cos(emb)

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)  # (B, 1, T, D)
    sin = sin[position_ids].unsqueeze(1)  # (B, 1, T, D)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# --- Attention and MLP Modules ---
class ScalableTokenizedAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_query_parameter_tokens, num_key_value_parameter_tokens):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.num_query_parameter_tokens = num_query_parameter_tokens
        self.num_key_value_parameter_tokens = num_key_value_parameter_tokens

        self.query_parameter_embeddings = nn.Parameter(torch.randn(num_query_parameter_tokens, hidden_size))
        self.key_parameter_embeddings = nn.Parameter(torch.randn(num_key_value_parameter_tokens, hidden_size))
        self.value_parameter_embeddings = nn.Parameter(torch.randn(num_key_value_parameter_tokens, hidden_size))

        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.zeros_(self.o_proj.bias)

    def forward(self, hidden_states, attention_mask, position_ids):
      batch_size, seq_len, _ = hidden_states.shape

      # Expand parameter tokens
      query_parameter_tokens = self.query_parameter_embeddings.expand(batch_size, self.num_query_parameter_tokens, self.hidden_size)
      key_parameter_tokens = self.key_parameter_embeddings.expand(batch_size, self.num_key_value_parameter_tokens, self.hidden_size)
      value_parameter_tokens = self.value_parameter_embeddings.expand(batch_size, self.num_key_value_parameter_tokens, self.hidden_size)

      q = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
      k = key_parameter_tokens.view(batch_size, self.num_key_value_parameter_tokens, self.num_heads, self.head_dim).transpose(1, 2)
      v = value_parameter_tokens.view(batch_size, self.num_key_value_parameter_tokens, self.num_heads, self.head_dim).transpose(1, 2)

      # Compute attention scores
      scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=hidden_states.device))
      
      # Debugging shapes uncomment the print statements below,there are 3 of them.
      #print(f"Scores shape: {scores.shape}")  # Shape: [batch_size, num_heads, seq_len, num_key_value_parameter_tokens]
      
      # Adjust attention mask dimensions
      if attention_mask is not None:
          #print(f"Attention mask shape before unsqueeze: {attention_mask.shape}")
          
          attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # Align dimensions
  
          # Truncate or pad attention mask to match `scores`
          target_dim = scores.shape[-1]
          
          if attention_mask.shape[-1] < target_dim:
              pad_size = target_dim - attention_mask.shape[-1]
              attention_mask = F.pad(attention_mask, (0, pad_size), value=float("-inf"))
          elif attention_mask.shape[-1] > target_dim:
              attention_mask = attention_mask[..., :target_dim]    

          #print(f"Attention mask shape after padding: {attention_mask.shape}")
          scores = scores + attention_mask

      # Compute attention weights
      attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)

      # Compute attention output
      attn_output = torch.matmul(attn_weights, v)
      attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
      attn_output = self.o_proj(attn_output)

      return attn_output

class ScalableTokenizedMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# --- Transformer Block and Model ---
class ScalableTokenizedTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, num_query_parameter_tokens, num_key_value_parameter_tokens):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = ScalableTokenizedAttention(hidden_size, num_heads, num_query_parameter_tokens, num_key_value_parameter_tokens)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = ScalableTokenizedMLP(hidden_size, intermediate_size)

    def forward(self, hidden_states, attention_mask, position_ids):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output = self.self_attn(hidden_states, attention_mask, position_ids)
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        return hidden_states

class ScalableTokenizedModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, intermediate_size,
                 num_query_parameter_tokens, num_key_value_parameter_tokens, max_position_embeddings):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.num_query_parameter_tokens = num_query_parameter_tokens
        self.num_key_value_parameter_tokens = num_key_value_parameter_tokens
        self.max_position_embeddings = max_position_embeddings

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            ScalableTokenizedTransformerBlock(hidden_size, num_heads, intermediate_size,
                                             num_query_parameter_tokens, num_key_value_parameter_tokens)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.rotary_emb = RotaryEmbedding(self.hidden_size // self.num_heads, base=10000)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        inputs_embeds = self.embedding(input_ids)

        if attention_mask is not None:
            attention_mask = (attention_mask.float().masked_fill(attention_mask == 0, float("-inf")).masked_fill(attention_mask == 1, float(0.0)))

        position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)

        sin, cos = self.rotary_emb(seq_len, input_ids.device)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def generate_config(self, save_path):
        config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_size": self.intermediate_size,
            "num_query_parameter_tokens": self.num_query_parameter_tokens,
            "num_key_value_parameter_tokens": self.num_key_value_parameter_tokens,
            "max_position_embeddings": self.max_position_embeddings
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(**config)

	  # --- Inference and Scaling Functions ---
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # Prepare inputs for the next generation step
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @torch.no_grad()

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_length=100,
        min_length=0,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.0,
        pad_token_id=None,
        eos_token_id=None,
        **kwargs
    ):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        if pad_token_id is None:
            pad_token_id = 128001  # Assuming 128001, 128008,128009 is pad_token_id
        if eos_token_id is None:
            eos_token_id = 128001 # Assuming 128001, 128008,128009 is eos_token_id

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        cur_len = input_ids.shape[1]

        while cur_len < max_length:
            # Prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids,
                attention_mask=attention_mask,
            )

            # Forward pass
            outputs = self.forward(**model_inputs)
            next_token_logits = outputs[:, -1, :]

            # Apply temperature
            next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(input_ids[i].tolist()):
                        next_token_logits[i, previous_token] /= repetition_penalty

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = float('-inf')

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences.unsqueeze(-1) + pad_token_id * (1 - unfinished_sequences.unsqueeze(-1))

            # Update input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_tokens)], dim=-1)

            # Update unfinished sequences
            unfinished_sequences = unfinished_sequences.mul(next_tokens.ne(eos_token_id).squeeze())

            cur_len += 1

            # Early stopping if all sequences are finished
            if unfinished_sequences.max() == 0:
                break

        return input_ids

def train_model(
    model: ScalableTokenizedModel,
    train_dataloader: DataLoader,
    config: TrainingConfig,
    eval_dataloader: Optional[DataLoader] = None,
    tokenizer = None
) -> Dict[str, Union[float, list]]:
    """
    Train the ScalableTokenizedModel.
    
    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        config: TrainingConfig object containing training parameters
        eval_dataloader: Optional DataLoader for evaluation
        tokenizer: Optional tokenizer for generating text samples during evaluation
    
    Returns:
        Dictionary containing training metrics
    """
    device = next(model.parameters()).device
    model.train()
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Initialize learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config.max_steps)
    
    # Initialize loss function
    criterion = CrossEntropyLoss()
    
    # Initialize tracking variables
    global_step = 0
    total_loss = 0
    best_eval_loss = float('inf')
    training_metrics = {
        'train_losses': [],
        'eval_losses': [],
        'learning_rates': []
    }
    
    # Initialize wandb if requested
    if config.use_wandb:
        wandb.init(project="scalable_tokenized_model", config=vars(config))
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    progress_bar = tqdm(total=config.max_steps, desc="Training")
    
    try:
        while global_step < config.max_steps:
            epoch_loss = 0
            for batch_idx, batch in enumerate(train_dataloader):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask)
                
                # Calculate loss
                # Shift predictions and labels for causal LM
                shift_logits = outputs[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Scale loss if using gradient accumulation
                loss = loss / config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    
                    # Optimizer step
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Update tracking variables
                    global_step += 1
                    total_loss += loss.item() * config.gradient_accumulation_steps
                    epoch_loss += loss.item() * config.gradient_accumulation_steps
                    
                    # Log metrics
                    if config.use_wandb:
                        wandb.log({
                            'train_loss': loss.item() * config.gradient_accumulation_steps,
                            'learning_rate': scheduler.get_last_lr()[0],
                            'global_step': global_step
                        })
                    
                    training_metrics['train_losses'].append(loss.item() * config.gradient_accumulation_steps)
                    training_metrics['learning_rates'].append(scheduler.get_last_lr()[0])
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                    
                    # Evaluation
                    if global_step % config.eval_steps == 0 and eval_dataloader is not None:
                        eval_loss = evaluate_model(model, eval_dataloader, criterion, device)
                        training_metrics['eval_losses'].append(eval_loss)
                        
                        if eval_loss < best_eval_loss:
                            best_eval_loss = eval_loss
                            # Save best model
                            torch.save({
                                'step': global_step,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': best_eval_loss,
                            }, os.path.join(config.output_dir, 'best_model.pth'))
                        
                        if config.use_wandb:
                            wandb.log({'eval_loss': eval_loss})
                        
                        # Generate sample text if tokenizer is provided
                        if tokenizer is not None:
                            sample_text = generate_sample(model, tokenizer, device)
                            if config.use_wandb:
                                wandb.log({'sample_text': sample_text})
                    
                    # Save checkpoint
                    if global_step % config.save_steps == 0:
                        torch.save({
                            'step': global_step,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'loss': total_loss / global_step,
                        }, os.path.join(config.output_dir, f'checkpoint_{global_step}.pth'))
                
                if global_step >= config.max_steps:
                    break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        torch.save({
            'step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': total_loss / global_step,
        }, os.path.join(config.output_dir, 'interrupted_checkpoint.pth'))
    
    progress_bar.close()
    if config.use_wandb:
        wandb.finish()
    
    return training_metrics

# Helper function for inference
def inference(model, tokenizer, prompt, device="cuda", **kwargs):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Set default generation parameters if not provided
    generation_kwargs = {
        "max_length": kwargs.get("max_length", 100),
        "min_length": kwargs.get("min_length", 0),
        "temperature": kwargs.get("temperature", 0.7),
        "top_k": kwargs.get("top_k", 50),
        "top_p": kwargs.get("top_p", 0.9),
        "repetition_penalty": kwargs.get("repetition_penalty", 1.0),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
    
def scale_model(model, scaled_num_query_parameter_tokens, scaled_num_key_value_parameter_tokens):
    original_num_query_parameter_tokens = model.num_query_parameter_tokens
    original_num_key_value_parameter_tokens = model.num_key_value_parameter_tokens

    new_model = ScalableTokenizedModel(
        vocab_size=model.vocab_size,
        hidden_size=model.hidden_size,
        num_layers=model.num_layers,
        num_heads=model.num_heads,
        intermediate_size=model.intermediate_size,
        num_query_parameter_tokens=scaled_num_query_parameter_tokens,
        num_key_value_parameter_tokens=scaled_num_key_value_parameter_tokens,
        max_position_embeddings=model.max_position_embeddings
    )

    # Copy existing weights and initialize new ones
    with torch.no_grad():
        for i in range(model.num_layers):
            # Copy attention weights
            new_model.layers[i].self_attn.query_parameter_embeddings[:original_num_query_parameter_tokens, :] = model.layers[i].self_attn.query_parameter_embeddings
            nn.init.xavier_normal_(new_model.layers[i].self_attn.query_parameter_embeddings[original_num_query_parameter_tokens:, :])

            new_model.layers[i].self_attn.key_parameter_embeddings[:original_num_key_value_parameter_tokens, :] = model.layers[i].self_attn.key_parameter_embeddings
            nn.init.xavier_normal_(new_model.layers[i].self_attn.key_parameter_embeddings[original_num_key_value_parameter_tokens:, :])

            new_model.layers[i].self_attn.value_parameter_embeddings[:original_num_key_value_parameter_tokens, :] = model.layers[i].self_attn.value_parameter_embeddings
            nn.init.xavier_normal_(new_model.layers[i].self_attn.value_parameter_embeddings[original_num_key_value_parameter_tokens:, :])

            # Copy other weights (assuming the architecture remains the same)
            new_model.layers[i].input_layernorm.weight.copy_(model.layers[i].input_layernorm.weight)
            new_model.layers[i].post_attention_layernorm.weight.copy_(model.layers[i].post_attention_layernorm.weight)
            new_model.layers[i].mlp.gate_proj.weight.copy_(model.layers[i].mlp.gate_proj.weight)
            new_model.layers[i].mlp.gate_proj.bias.copy_(model.layers[i].mlp.gate_proj.bias)
            new_model.layers[i].mlp.down_proj.weight.copy_(model.layers[i].mlp.down_proj.weight)
            new_model.layers[i].mlp.down_proj.bias.copy_(model.layers[i].mlp.down_proj.bias)
            new_model.layers[i].mlp.up_proj.weight.copy_(model.layers[i].mlp.up_proj.weight)
            new_model.layers[i].mlp.up_proj.bias.copy_(model.layers[i].mlp.up_proj.bias)

        new_model.embedding.weight.copy_(model.embedding.weight)
        new_model.norm.weight.copy_(model.norm.weight)
        new_model.lm_head.weight.copy_(model.lm_head.weight)
        new_model.lm_head.bias.copy_(model.lm_head.bias)

    return new_model

if __name__ == '__main__':
    # --- Initial Model Setup ---
    vocab_size = 128256
    hidden_size = 256
    num_layers = 2
    num_heads = 4
    intermediate_size = 512
    num_query_parameter_tokens = 64
    num_key_value_parameter_tokens = 128
    max_position_embeddings = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("DeepAutoAI/Explore_Llama-3.2-1B-Inst_v1.1") # Or any suitable tokenizer

    model = ScalableTokenizedModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        num_query_parameter_tokens=num_query_parameter_tokens,
        num_key_value_parameter_tokens=num_key_value_parameter_tokens,
        max_position_embeddings=max_position_embeddings
    ).to(device)

    # --- Load or Save Initial Model (Optional) ---
    config_save_path = "my_tokenformer_config"
    os.makedirs(config_save_path, exist_ok=True)
    model.generate_config(config_save_path)
    # torch.save(model.state_dict(), os.path.join(config_save_path, "model.pth"))
    # model.load_state_dict(torch.load(os.path.join(config_save_path, "model.pth")))

    # --- Inference with Initial Model ---
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    output = inference(model, tokenizer, "Write a short story about a friendly robot.", 
                      max_length=200,
                      temperature=0.7,
                      top_k=50,
                      top_p=0.9)
    print(output)

    # --- Ask User to Scale ---
    scale_choice = input("Would you like to scale the model? (yes/no): ").lower()
    if scale_choice == "yes":
        scaled_num_query_parameter_tokens = int(input("Enter the new number of query parameter tokens: "))
        scaled_num_key_value_parameter_tokens = int(input("Enter the new number of key-value parameter tokens: "))

        # --- Scale the Model ---
        scaled_model = scale_model(model, scaled_num_query_parameter_tokens, scaled_num_key_value_parameter_tokens).to(device)
        print("Model scaled successfully!")

        # --- Inference with Scaled Model ---
        scaled_output = inference(scaled_model, tokenizer, initial_prompt, device)
        print(f"Scaled Model Output:\n{scaled_output}")

        # --- Save Scaled Model (Optional) ---
        scaled_config_save_path = "my_scaled_tokenformer_config"
        os.makedirs(scaled_config_save_path, exist_ok=True)
        scaled_model.generate_config(scaled_config_save_path)
        # torch.save(scaled_model.state_dict(), os.path.join(scaled_config_save_path, "model.pth"))
    else:
        print("Not scaling the model.")

    # For loading from a local text file:
    train_dataloader, eval_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        file_path="/content/pokemon.txt",
        batch_size=2,
        max_length=1024,
        cache_dir="./cache"
    )
    '''
    # Or for loading from HuggingFace:
    train_dataloader, eval_dataloader = create_dataloaders(
        tokenizer=tokenizer,
        dataset_name="wikitext",  # or any other HuggingFace dataset
        batch_size=32,
        max_length=1024,
        cache_dir="./cache"
    )
    '''
    # Create training configuration
    config = TrainingConfig(
      learning_rate=5e-4,
      batch_size=2,
      max_steps=10000,
      eval_steps=1000,
      save_steps=5000,
      use_wandb=False  # Set to True if you want to use Weights & Biases
    )

    # Train the model
    metrics = train_model(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        tokenizer=tokenizer
    )
