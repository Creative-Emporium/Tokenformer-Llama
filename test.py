import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
from transformers import AutoTokenizer

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
        query_parameter_tokens = self.query_parameter_embeddings.expand(batch_size, self.num_query_parameter_tokens, self.hidden_size)
        key_parameter_tokens = self.key_parameter_embeddings.expand(batch_size, self.num_key_value_parameter_tokens, self.hidden_size)
        value_parameter_tokens = self.value_parameter_embeddings.expand(batch_size, self.num_key_value_parameter_tokens, self.hidden_size)

        q = hidden_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = key_parameter_tokens.view(batch_size, self.num_key_value_parameter_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        v = value_parameter_tokens.view(batch_size, self.num_key_value_parameter_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE if you want to incorporate positional information for input tokens
        # (Note: Parameter tokens don't inherently have a position in this formulation)
        # sin, cos = self.rotary_emb(seq_len, hidden_states.device)
        # q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32, device=hidden_states.device))

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
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
def inference(model, tokenizer, prompt, device="cuda"):
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, temperature=0.7)  # Adjust max_length and temperature as needed
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
    vocab_size = 32000
    hidden_size = 512
    num_layers = 6
    num_heads = 8
    intermediate_size = 1024
    num_query_parameter_tokens = 64
    num_key_value_parameter_tokens = 128
    max_position_embeddings = 2048
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct") # Or any suitable tokenizer

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
    initial_prompt = "Write a short story about a friendly robot."
    output = inference(model, tokenizer, initial_prompt, device)
    print(f"Initial Model Output:\n{output}")

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
