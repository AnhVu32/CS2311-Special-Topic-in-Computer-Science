"""Huấn luyện executor bằng RL (GRPO + LoRA), dùng đề trong KB và harness của project để chấm thưởng.

- data.py:       đề KB -> prompt cùng định dạng MBPP+ của benchmark, test để chấm, khử ô nhiễm, chia train/val
- rewards.py:    thưởng = tỉ lệ test qua trong harness (+ thưởng định dạng), tuỳ chọn co giãn kiểu ADPO theo p_hat
- hf_backend.py: sinh lời giải theo lô bằng transformers (dùng cho p_hat, benchmark và chấm val)

Script: tools/grpo_executor.py (prepare / train / merge / score), tools/hf_generate.py (MBPP+ bằng transformers).
"""
