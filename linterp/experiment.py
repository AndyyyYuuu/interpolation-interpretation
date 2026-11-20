import os
import torch
from nnsight import LanguageModel

_HF_LOGIN_DONE = False


def _hf_login_if_needed() -> None:
    global _HF_LOGIN_DONE
    if _HF_LOGIN_DONE:
        return
    token = os.getenv("HF_TOKEN")
    if token is None:
        return
    from huggingface_hub import get_token, login
    if get_token() == token:
        _HF_LOGIN_DONE = True
        return
    login(token=token)
    _HF_LOGIN_DONE = True


class EmbedLerp:
    """Interpolates between two token embeddings at a single position in a prompt."""

    def __init__(self, model_name: str, prompt: str, tok_0: str, tok_1: str, classes: list[str]):
        _hf_login_if_needed()
        self.model = LanguageModel(model_name, device_map="auto", dispatch=True, torch_dtype=torch.float32)
        self.name = model_name

        tokenizer = self.model.tokenizer
        embed_weight = self.model.model.embed_tokens.weight.detach()

        parts = prompt.split("{}")
        assert len(parts) == 2, "prompt must contain exactly one {}"

        prefix_ids = tokenizer(parts[0], return_tensors="pt").input_ids[0]       # includes BOS
        suffix_ids = tokenizer(parts[1], return_tensors="pt").input_ids[0, 1:]   # skip BOS
        tok_0_id   = tokenizer(tok_0, return_tensors="pt").input_ids[0, 1]       # skip BOS
        tok_1_id   = tokenizer(tok_1, return_tensors="pt").input_ids[0, 1]       # skip BOS

        self.prefix_embeds = embed_weight[prefix_ids]   # (prefix_len, hidden)
        self.suffix_embeds = embed_weight[suffix_ids]   # (suffix_len, hidden)
        self.embed_0 = embed_weight[tok_0_id]           # (hidden,)
        self.embed_1 = embed_weight[tok_1_id]           # (hidden,)

        self.tracked_ids = [tokenizer(c, return_tensors="pt").input_ids[0, -1].item() for c in classes]
        self.seq_len = prefix_ids.shape[0] + 1 + suffix_ids.shape[0]

    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(self, weight: float) -> torch.Tensor:
        lerped = torch.lerp(self.embed_0, self.embed_1, weight)
        total = torch.cat([
            self.prefix_embeds, lerped.unsqueeze(0), self.suffix_embeds
        ], dim=0).unsqueeze(0).to(device=self.device, dtype=self.embed_0.dtype)  # (1, seq_len, hidden)

        dummy_ids = torch.zeros((1, self.seq_len), dtype=torch.long, device=self.device)
        with self.model.trace(dummy_ids):
            self.model.model.embed_tokens.output = total
            logits = self.model.output.save()

        probs = torch.softmax(logits.logits[0, -1].float(), dim=-1)
        return probs[self.tracked_ids].cpu()


class HiddenLerp:
    """Interpolates hidden states between two prompts at a given layer, sweeping across layers."""

    def __init__(self, model_name: str, str_0: str, str_1: str, classes: list[str]):
        _hf_login_if_needed()
        self.model = LanguageModel(model_name, device_map="auto", dispatch=True, torch_dtype=torch.float32)
        self.name = model_name
        self.n_layers = self.model.config.num_hidden_layers

        tokenizer = self.model.tokenizer

        ids_0 = tokenizer(str_0, return_tensors="pt").input_ids.to(self.device)
        ids_1 = tokenizer(str_1, return_tensors="pt").input_ids.to(self.device)
        assert ids_0.shape == ids_1.shape, "str_0 and str_1 must tokenize to the same length"

        self.tracked_ids = [tokenizer(c, return_tensors="pt").input_ids[0, -1].item() for c in classes]
        self._dummy_ids = torch.zeros_like(ids_0)

        with self.model.trace(ids_0):
            h0 = [self.model.model.layers[i].output[0].save() for i in range(self.n_layers)]

        with self.model.trace(ids_1):
            h1 = [self.model.model.layers[i].output[0].save() for i in range(self.n_layers)]

        self.h0 = h0  # list of (1, seq_len, hidden) tensors, one per layer
        self.h1 = h1

    @property
    def device(self) -> torch.device:
        return self.model.device

    def forward(self, weight: float, layer: int) -> torch.Tensor:
        lerped = torch.lerp(self.h0[layer], self.h1[layer], weight)
        with self.model.trace(self._dummy_ids):
            self.model.model.layers[layer].output = lerped
            logits = self.model.output.save()

        probs = torch.softmax(logits.logits[0, -1].float(), dim=-1)
        return probs[self.tracked_ids].cpu()
