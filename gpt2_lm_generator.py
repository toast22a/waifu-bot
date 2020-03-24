import os

import torch
import torch.nn.functional as F

from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel
)

class GPT2LMGenerator:
    def __init__(self, model_dir, device='cpu'):
        assert(os.path.isdir(model_dir))

        self.model_dir = model_dir
        self.device = device
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def sample(self, context_raw, max_context=32, min_generation=0, max_generation=32,
            temperature=0.7, repetition_penalty=1.0, stop_tokens=['<|endoftext|>']):
        stop_token_indices = [self.tokenizer.encode(stop_token)[0] for stop_token in stop_tokens]

        context = self.tokenizer.encode(context_raw)[-max_context:]
        context = torch.tensor(context, dtype=torch.long, device=self.device)
        context = context.unsqueeze(0)

        max_past = min(max_context+max_generation,
                self.tokenizer.max_len_single_sentence)
        next_token = None
        past = None
        generated = context
        with torch.no_grad():
            for _ in range(max_generation):
                outputs = self.model(next_token if next_token else generated, past=past)

                if temperature == 0: # greedy
                    next_token_logits = outputs[0][:, -1, :]
                else:
                    next_token_logits = outputs[0][:, -1, :] / temperature
                
                if _ < min_generation:
                    for x in stop_token_indices:
                        next_token_logits[0, x] = -float('Inf')

                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > 0.9
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')

                # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
                for _ in set(generated[0].tolist()):
                    next_token_logits[0, _] /= repetition_penalty

                if temperature == 0: # greedy
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                else:
                    next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

                generated = torch.cat((generated, next_token), dim=1)
                if next_token.item() in stop_token_indices: break

                past = [p[..., -max_past:, :] for p in outputs[1]]

        generated = generated.to('cpu')
        out = generated[:, context.size()[1]:].tolist()
        out_decoded = self.tokenizer.decode(out[0])
        out_decoded = out_decoded[:min([x if x >= 0 else len(out_decoded) for x in
            [out_decoded.find(stop_token) for stop_token in stop_tokens]],
            default=len(out_decoded))]

        return out_decoded
