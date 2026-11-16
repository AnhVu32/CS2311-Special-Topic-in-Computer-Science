"""Chạy assert của một bài trong tiến trình con cô lập (dùng cho tập dev của run_benchmark).

Lớp mỏng trên agent0/execution/harness.py (cùng harness với pipeline sinh KB): lời giải nạp như
module, exit() tính là FAIL, từng assert chạy riêng, env sạch, giới hạn tài nguyên, chặn mạng.
"""
from typing import Dict, List

from agent0.execution.harness import run_harness


def run_tests_isolated(
    solution: str,
    tests: List[str],
    setup: str = "",
    timeout: float = 10.0,
    memory_mb: int = 2048,
) -> Dict:
    """Trả về {'passed', 'n_tests', 'n_ok', 'error', 'fails'}; passed chỉ True khi mọi assert đều qua."""
    res = run_harness(solution, tests, setup, timeout=timeout, memory_mb=memory_mb)
    error = res.get("load_error")
    if not error and not res.get("referenced", True):
        error = "test không gọi tới hàm/lớp nào của lời giải"
    return {
        "passed": bool(res.get("passed")),
        "n_tests": res.get("n_asserts", 0),
        "n_ok": res.get("n_ok", 0),
        "error": error,
        "fails": [
            {"test": f.get("test"), "error": f.get("error")}
            for f in (res.get("failures", []) + res.get("statement_errors", []))
        ],
    }
