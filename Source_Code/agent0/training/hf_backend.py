"""Sinh lời giải theo lô bằng transformers (không cần vLLM / Ollama): dùng cho p_hat, benchmark và chấm val.

Trên T4: fp16 (T4 không có bf16), attention "sdpa" (không có FlashAttention-2). Trên CPU: fp32 (chỉ để thử).
Giải mã tham lam đặt rõ do_sample=False, repetition_penalty=1.0 và bỏ top_k / top_p mặc định trong
generation_config của model, để base và model sau RL được sinh theo đúng một cách.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


def pick_device(device: Optional[str] = None) -> str:
    import torch

    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_dtype(device: str, dtype: str = "auto"):
    import torch

    if dtype == "auto":
        return torch.float16 if device.startswith("cuda") else torch.float32
    return {"float16": torch.float16, "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32,
            "bfloat16": torch.bfloat16, "bf16": torch.bfloat16}[dtype]


def load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


class HFGenerator:
    def __init__(self, model_path: str, adapter: Optional[str] = None, device: Optional[str] = None,
                 dtype: str = "auto", max_new_tokens: int = 1024):
        import torch
        from transformers import AutoModelForCausalLM

        self.device = pick_device(device)
        self.dtype = pick_dtype(self.device, dtype)
        self.tok = load_tokenizer(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=self.dtype, attn_implementation="sdpa")
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        self.model = model.to(self.device).eval()
        self.max_new_tokens = max_new_tokens
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tok.eos_token_id
        self.eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        if self.tok.eos_token_id is not None and self.tok.eos_token_id not in self.eos_ids:
            self.eos_ids.append(self.tok.eos_token_id)
        # transformers 4.x thay giá trị bằng mặc định (temperature=1.0, repetition_penalty=1.0) bằng giá trị trong
        # generation_config.json của model (Qwen: 0.7 / 1.1) — transformers#42762, đã sửa ở v5. Để mọi bản cho cùng
        # một cách sinh, generation_config của model chỉ giữ token đặc biệt; cấu hình sinh luôn truyền rõ ở generate().
        from transformers import GenerationConfig

        old = self.model.generation_config
        self.model.generation_config = GenerationConfig(
            bos_token_id=getattr(old, "bos_token_id", None), eos_token_id=self.eos_ids,
            pad_token_id=getattr(old, "pad_token_id", None) or self.tok.pad_token_id)
        self._torch = torch
        logger.info("HFGenerator: %s%s trên %s (%s)", model_path, f" + {adapter}" if adapter else "", self.device, self.dtype)

    def render(self, messages: List[dict]) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _gen_config(self, temperature: float, top_p: float, n: int, max_new_tokens: int):
        from transformers import GenerationConfig

        sample = temperature > 0
        return GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=sample,
            temperature=temperature if sample else None,
            top_p=top_p if sample else None,
            top_k=0 if sample else None,  # 0 = tắt top-k (như rollout của TRL); None thì lấy top_k=20 mặc định của Qwen
            repetition_penalty=1.0,
            num_return_sequences=n if sample else 1,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.eos_ids,
        )

    def generate(
        self,
        conversations: Sequence[List[dict]],
        n: int = 1,
        temperature: float = 0.0,
        top_p: float = 1.0,
        batch_size: int = 16,
        seed: int = 0,
        max_new_tokens: Optional[int] = None,
        progress: bool = True,
    ) -> List[List[str]]:
        """Trả về, cho mỗi hội thoại, n câu trả lời (giải mã tham lam thì n câu giống nhau)."""
        torch = self._torch
        n = max(1, n)
        per_batch = max(1, batch_size // (n if temperature > 0 else 1))
        texts = [self.render(c) for c in conversations]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # độ dài gần nhau -> ít padding
        out: List[List[str]] = [[] for _ in texts]
        cfg = self._gen_config(temperature, top_p, n, max_new_tokens or self.max_new_tokens)
        n_batches = (len(order) + per_batch - 1) // per_batch
        for b, start in enumerate(range(0, len(order), per_batch), 1):
            idx = order[start:start + per_batch]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            torch.manual_seed(seed + start)
            with torch.no_grad():
                gen = self.model.generate(**enc, generation_config=cfg)
            gen = gen[:, enc["input_ids"].shape[1]:]
            decoded = self.tok.batch_decode(gen, skip_special_tokens=True)
            per = len(decoded) // len(idx)
            for j, i in enumerate(idx):
                answers = decoded[j * per:(j + 1) * per]
                out[i] = answers if temperature > 0 else answers * n
            if progress and (b == n_batches or b % max(1, n_batches // 10) == 0):
                logger.info("sinh %s/%s lô (%s hội thoại)", b, n_batches, len(texts))
        return out
