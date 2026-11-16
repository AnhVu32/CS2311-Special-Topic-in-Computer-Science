import logging
from typing import Any, Dict, List

from agent0.execution.harness import format_report, run_harness

logger = logging.getLogger("agent0")


def record_tests(record: dict, use_challenge_tests: bool = True) -> List[str]:
    tests = list(record.get("test_list", []) or [])
    if use_challenge_tests and record.get("challenge_test_list"):
        tests.extend(record["challenge_test_list"])
    return tests


def build_verification_script(record: dict, solution_code: str, use_challenge_tests: bool = True) -> str:
    """Giữ lại để tương thích: script ghép như bản cũ (chỉ để xem/gỡ lỗi, không còn dùng để chấm)."""
    parts = []
    if record.get("test_setup_code"):
        parts.append(record["test_setup_code"])
    parts.append(solution_code)
    parts.extend(record_tests(record, use_challenge_tests))
    return "\n\n".join(parts)


def verify_solution(record: dict, solution_code: str, timeout: int = 30, use_challenge_tests: bool = True) -> Dict[str, Any]:
    """Chạy test của bài với lời giải trong harness cô lập (agent0/execution/harness.py).

    Giữ nguyên các khóa cũ (passed, returncode, stdout, stderr, script_path); 'stderr' giờ là báo cáo
    dễ đọc (assert nào sai, giá trị thực tế vs mong đợi) để đưa vào prompt chẩn đoán/sửa.
    'details' chứa kết quả đầy đủ của harness.
    """
    tests = record_tests(record, use_challenge_tests)
    logger.debug(
        "Verification start | task_id=%s timeout=%ss tests=%s", record.get("task_id"), timeout, len(tests),
    )
    res = run_harness(solution_code, tests, record.get("test_setup_code", "") or "", timeout=timeout)
    passed = bool(res.get("passed"))
    logger.info(
        "Verification %s | task_id=%s asserts=%s/%s%s",
        "PASS" if passed else "FAIL",
        record.get("task_id"),
        res.get("n_ok", 0),
        res.get("n_asserts", 0),
        " (timeout)" if res.get("timed_out") else "",
    )
    return {
        "passed": passed,
        "returncode": 0 if passed else (-1 if res.get("timed_out") else 1),
        "stdout": "",
        "stderr": "" if passed else format_report(res),
        "script_path": None,
        "details": {
            k: res.get(k)
            for k in (
                "loaded", "load_error", "n_asserts", "n_ok", "n_statements", "converted", "syntax_errors",
                "failures", "statement_errors", "referenced", "defined_names", "timed_out",
            )
        },
    }
