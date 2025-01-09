import os
import torch
import json
import argparse
from transformers import AutoTokenizer
from create_model import TokenFormer, load_json_config
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0):
    """
    Filters a distribution of logits using top-k and/or top-p.
    Args:
        logits (torch.Tensor): Logits for the distribution (usually the output of a model).
        top_k (int, optional): The number of top logits to keep. Defaults to 0 (no top-k).
        top_p (float, optional): The cumulative probability threshold for top-p. Defaults to 1.0 (no top-p).

    Returns:
        torch.Tensor: Filtered logits.
    """
    if top_k > 0:
        top_k = min(top_k, logits.size(-1))  # Safety check
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., [-1]]
        logits = logits.masked_fill(indices_to_remove, -float('inf'))

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Find indices to remove using top_p logic
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Gather indices to remove
        indices_to_remove = sorted_indices.gather(-1, sorted_indices_to_remove.nonzero()[:, 1].unsqueeze(0))
        
        # Apply the mask only if indices_to_remove is not empty
        if indices_to_remove.numel() > 0:
             # Create the mask using broadcasting, handling cases where indices_to_remove doesn't include all indices
              mask = torch.ones_like(logits, dtype = torch.bool)
              mask.scatter_(1, indices_to_remove, 0) # zero at indices to remove
              logits = logits.masked_fill(~mask, -float('inf'))
    return logits


class InferenceEngine:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device # store the device

    @torch.no_grad()
    def generate_text(self, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0, do_sample: bool = True) -> str:
       """
        Generates text based on the given prompt.

        Args:
           prompt (str): Input prompt for text generation.
           max_new_tokens (int, optional): Max number of new tokens to generate. Defaults to 50.
           temperature (float, optional): Sampling temperature. Defaults to 1.0.
           top_p (float, optional): Nucleus sampling top-p value. Defaults to 1.0.
           top_k (int, optional): Top-k sampling value. Defaults to 0.
           do_sample (bool, optional): Whether to use sampling (vs. greedy). Defaults to True.

        Returns:
           str: Generated text.
        """
       self.model.eval() # ensure model is in eval mode for inference
       input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device) # tokenize prompt and send to stored device
       generated_ids = [] # list to store generated token ids

       for _ in range(max_new_tokens):
           outputs = self.model(input_ids) # shape (batch_size, seq_len, vocab_size)
           next_token_logits = outputs[:, -1, :] # shape (batch_size, vocab_size) (takes last logit)

           if do_sample:
                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits/temperature
                
                # Apply Top-k and Top-p filtering
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                
                # sample from probability distribution
                next_token_probs = F.softmax(filtered_logits, dim = -1)
                next_token_id = torch.multinomial(next_token_probs, num_samples=1)
           else:
              next_token_id = torch.argmax(next_token_logits, dim = -1, keepdim=True) # greedy selection if not sampling

           generated_ids.append(next_token_id)
           input_ids = torch.cat((input_ids, next_token_id), dim = -1) # concatenate generated token id to input ids
           if next_token_id == self.tokenizer.eos_token_id:
               break # stop generation if end of sequence token is generated

       generated_ids = torch.cat(generated_ids, dim = -1)  #concatenate to tensor
       return self.tokenizer.decode(generated_ids[0], skip_special_tokens = True)

def main():
    parser = argparse.ArgumentParser(description="Inference script for TokenFormer")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model directory")
    parser.add_argument("--prompt", type=str, default="This is an example ", help="Prompt for text generation")
    parser.add_argument("--max_new_tokens", type = int, default = 50, help = "Maximum number of tokens to generate")
    parser.add_argument("--temperature", type = float, default = 1.0, help = "Sampling temperature")
    parser.add_argument("--top_p", type = float, default = 1.0, help="Nucleus Sampling, top p value")
    parser.add_argument("--top_k", type = int, default = 0, help="Top-k Sampling")
    parser.add_argument("--do_sample", action="store_true", help = "Use sampling for generation")
    args = parser.parse_args()

    # Load configurations
    config = load_json_config(os.path.join(args.model_path, "config.json"))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Initialize Model
    model = TokenFormer(
        vocab_size = config['vocab_size'],
        d_model = config['d_model'],
        n_layers = config['n_layers'],
        num_groups = config['num_groups'],
        expansion_factor = config['intermediate_size'] // config['d_model'],
        dropout = config['dropout'],
        max_seq_len = config['max_position_embeddings'],
        num_attention_heads = config['num_attention_heads'],
        qkv_slot_num = config['qkv_slot_num'],
        proj_slot_num = config['proj_slot_num'],
        activation_fn = config['hidden_act'],
        use_bias_in_attn_linear = config['use_bias_in_attn_linear'],
        rotary_pct = config['rotary_pct'],
        init_method = config['init_method'],
        output_layer_init_method = config['output_layer_init_method'],
        norm_activation_type = config.get('norm_activation_type', 'gelu'),
        initializer_range = config.get('initializer_range', 0.02),
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) #move model to device before initializing InferenceEngine

    # Load weights from saved model
    state_dict = torch.load(os.path.join(args.model_path, 'model.pth'), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Initialize Inference Engine
    inference_engine = InferenceEngine(model, tokenizer, device) # Pass the device object

    # Generate text
    generated_text = inference_engine.generate_text(
        prompt=args.prompt, 
        max_new_tokens=args.max_new_tokens, 
        temperature = args.temperature,
        top_p = args.top_p,
        top_k = args.top_k,
        do_sample = args.do_sample
        )
    print(f"\nGenerated text: {generated_text}")

if __name__ == "__main__":
    main()
