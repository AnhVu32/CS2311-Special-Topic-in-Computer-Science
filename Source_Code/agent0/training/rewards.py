"""Thưởng cho GRPO của executor, chấm bằng harness của project (cách ly tiến trình, giới hạn CPU/RAM, chặn mạng).

    r = tỉ lệ assert qua            (--reward fraction, mặc định)   hoặc   1[qua hết test]   (--reward binary)
      + format_bonus * 1[có khối code và code nạp được]   (code bị cắt cụt / lỗi cú pháp không được thưởng)
    ADPO (xấp xỉ): r *= s(p_hat),  s(p_hat) = s_min + (1 - s_min) * p_hat

Agent0 dùng nhãn giả theo đa số; ở đây có test đã kiểm chứng nên thưởng lấy thẳng từ việc chạy test.
Với scale_rewards="none" của TRL, advantage = r - mean(r) của nhóm, nên nhân r với s(p_hat) của đề
chính là nhân advantage — đúng phần "co giãn advantage theo p_hat" của ADPO (phần ε_high theo từng đề không có).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from agent0.core.utils import extract_code_block, strip_think_blocks
from agent0.execution.harness import run_harness

REWARD_MODES = ("fraction", "binary")


def completion_text(completion: Any) -> str:
    """TRL đưa completion dạng chuỗi hoặc danh sách message (prompt hội thoại)."""
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content") or "")
    if isinstance(completion, (list, tuple)):
        return "".join(str(m.get("content") or "") for m in completion
                       if isinstance(m, dict) and m.get("role", "assistant") == "assistant")
    return str(completion)


def solution_from_text(text: str) -> tuple:
    """(lời giải, có khối code không) — cùng cách run_benchmark.extract_solution lấy lời giải."""
    body = strip_think_blocks(text or "")
    code = extract_code_block(body)
    return ((code if code is not None else body).strip(), code is not None)


def score_completion(text: str, tests: List[str], setup: str = "", timeout: float = 10.0, memory_mb: int = 1024) -> Dict:
    solution, has_code = solution_from_text(text)
    out = {"pass_frac": 0.0, "passed": False, "has_code": has_code, "code_ok": False, "n_ok": 0, "n_asserts": 0,
           "error": None}
    if not solution:
        out["error"] = "empty"
        return out
    res = run_harness(solution, list(tests or []), setup=setup or "", timeout=timeout, memory_mb=memory_mb)
    n_asserts = int(res.get("n_asserts") or 0)
    n_ok = int(res.get("n_ok") or 0)
    loaded = bool(res.get("loaded")) and not res.get("timed_out")
    valid = loaded and bool(res.get("referenced", True))
    out.update({
        "code_ok": has_code and loaded,
        "n_ok": n_ok, "n_asserts": n_asserts, "passed": bool(res.get("passed")),
        "pass_frac": (n_ok / n_asserts) if (valid and n_asserts) else 0.0,
        "error": None if valid else (res.get("load_error") or ("timeout" if res.get("timed_out") else "not_referenced")),
    })
    return out


def adpo_scale(p_hat: Optional[float], s_min: float = 0.5) -> float:
    """s(p_hat) tăng theo p_hat như ADPO: đề model còn ít tự tin (p_hat thấp) -> bước cập nhật nhỏ hơn."""
    if p_hat is None:
        return 1.0
    try:
        p = float(p_hat)
    except (TypeError, ValueError):
        return 1.0
    if p != p:  # NaN
        return 1.0
    return s_min + (1.0 - s_min) * min(1.0, max(0.0, p))


class HarnessReward:
    """Hàm thưởng cho trl.GRPOTrainer: nhận cột `tests`, `setup`, `p_hat` của dataset qua kwargs."""

    def __init__(
        self,
        mode: str = "fraction",
        format_bonus: float = 0.1,
        timeout: float = 10.0,
        memory_mb: int = 1024,
        workers: int = 8,
        adpo: bool = False,
        s_min: float = 0.5,
    ):
        if mode not in REWARD_MODES:
            raise ValueError(f"reward mode không hợp lệ: {mode} (chọn {REWARD_MODES})")
        self.__name__ = "harness_reward"  # TRL dùng tên này khi ghi log
        self.mode = mode
        self.format_bonus = format_bonus
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.workers = max(1, workers)
        self.adpo = adpo
        self.s_min = s_min
        self.history: List[Dict] = []

    def __call__(self, prompts=None, completions=None, tests=None, setup=None, p_hat=None, **kwargs) -> List[float]:
        completions = list(completions or [])
        n = len(completions)
        tests = list(tests) if tests is not None else [[] for _ in range(n)]
        setup = list(setup) if setup is not None else ["" for _ in range(n)]
        p_hat = list(p_hat) if p_hat is not None else [None for _ in range(n)]
        texts = [completion_text(c) for c in completions]

        def one(i: int) -> Dict:
            return score_completion(texts[i], tests[i], setup[i] or "", self.timeout, self.memory_mb)

        with ThreadPoolExecutor(max_workers=min(self.workers, max(1, n))) as ex:
            scores = list(ex.map(one, range(n)))
        rewards = []
        for i, s in enumerate(scores):
            base = s["pass_frac"] if self.mode == "fraction" else float(s["passed"])
            r = base + (self.format_bonus if s["code_ok"] else 0.0)
            if self.adpo:
                r *= adpo_scale(p_hat[i], self.s_min)
            rewards.append(float(r))
        if n:
            self.history.append({
                "n": n,
                "pass_frac": round(sum(s["pass_frac"] for s in scores) / n, 4),
                "all_pass": round(sum(s["passed"] for s in scores) / n, 4),
                "has_code": round(sum(s["has_code"] for s in scores) / n, 4),
                "code_ok": round(sum(s["code_ok"] for s in scores) / n, 4),
                "reward": round(sum(rewards) / n, 4),
            })
        return rewards
