import ast
import hashlib
import json
import logging
import re
import textwrap
from typing import List, Optional, Set, Tuple

from agent0.core.utils import load_jsonl_or_json, safe_strip_text, resolve_existing_path

logger = logging.getLogger("agent0")

# Đề tự sinh phải có ít nhất chừng này assert (prompt sinh đề yêu cầu 3-5), ít hơn thì test quá yếu.
MIN_SYNTHETIC_ASSERTS = 3


def load_mbpp_plus_ids() -> Set[int]:
    """Số hiệu (task_id dạng số) của 378 bài MBPP+ — dùng để loại khỏi dữ liệu seed, tránh lộ đề test."""
    from agent0.core.config import MBPP_PLUS_TASK_IDS_FILE

    data = json.loads(MBPP_PLUS_TASK_IDS_FILE.read_text(encoding="utf-8"))
    return {int(t["task_id"].split("/")[1]) for t in data["tasks"]}


def normalize_test_lines(tests: List[str]) -> Tuple[List[str], int]:
    """Chuẩn hoá các dòng test do LLM sinh.

    - bỏ dòng không parse được và dòng chỉ là literal (vd chuỗi dict `{'input': ..., 'expected': ...}`);
    - đổi dòng so sánh trần `f(x) == y` thành `assert f(x) == y`;
    - giữ dòng lệnh thường (vd `s = Stack()`) vì test có trạng thái cần chúng.
    Trả về (các dòng giữ lại, số dòng có assert).
    """
    kept: List[str] = []
    n_assert = 0
    for raw in tests:
        line = (raw or "").strip()
        if not line:
            continue
        try:
            mod = ast.parse(line)
        except SyntaxError:
            continue
        body = mod.body
        if len(body) == 1 and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Compare):
                line = "assert " + line
                n_assert += 1
            elif isinstance(value, (ast.Constant, ast.Dict, ast.List, ast.Tuple, ast.Set)):
                continue
        elif any(isinstance(n, ast.Assert) for n in ast.walk(mod)):
            n_assert += 1
        kept.append(line)
    return kept, n_assert

def normalize_mbpp_record(
    raw: dict,
    prompt_key: str = "text",
    code_key: str = "code",
    test_key: str = "test_list",
    setup_key: str = "test_setup_code",
) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    prompt = raw.get(prompt_key)
    if prompt is None:
        for k in ("prompt", "question", "text", "instruction"):
            if raw.get(k) is not None:
                prompt = raw.get(k)
                break

    code = raw.get(code_key)
    if code is None:
        for k in ("code", "answer", "solution"):
            if raw.get(k) is not None:
                code = raw.get(k)
                break

    tests = raw.get(test_key)
    if tests is None:
        for k in ("test_list", "tests", "assertions"):
            if raw.get(k) is not None:
                tests = raw.get(k)
                break

    setup_code = raw.get(setup_key)
    if setup_code is None:
        setup_code = raw.get("test_setup_code", "")

    challenge_tests = raw.get("challenge_test_list", [])

    if prompt is None or code is None or tests is None:
        return None

    if isinstance(tests, str):
        tests = [x for x in tests.split("\n") if x.strip()]
    if isinstance(challenge_tests, str):
        challenge_tests = [x for x in challenge_tests.split("\n") if x.strip()]

    if not isinstance(tests, list):
        tests = [str(tests)]
    if not isinstance(challenge_tests, list):
        challenge_tests = [str(challenge_tests)] if challenge_tests else []

    record = {
        "task_id": raw.get("task_id"),
        "prompt": safe_strip_text(prompt),
        "reference_solution": safe_strip_text(code),
        "test_setup_code": safe_strip_text(setup_code),
        "test_list": [safe_strip_text(t) for t in tests if safe_strip_text(t)],
        "challenge_test_list": [safe_strip_text(t) for t in challenge_tests if safe_strip_text(t)],
        "source_file": raw.get("source_file", "mbpp.jsonl"),
        "raw": raw,
    }
    return record


def load_mbpp_records(
    data_file: str,
    prompt_key: str = "text",
    code_key: str = "code",
    test_key: str = "test_list",
    setup_key: str = "test_setup_code",
    limit: Optional[int] = None,
) -> List[dict]:
    path = resolve_existing_path(data_file)
    raw_items = load_jsonl_or_json(path)
    records: List[dict] = []
    for raw in raw_items:
        rec = normalize_mbpp_record(raw, prompt_key=prompt_key, code_key=code_key, test_key=test_key, setup_key=setup_key)
        if rec:
            records.append(rec)
        if limit and len(records) >= limit:
            break

    if not records:
        raise ValueError(
            f"No valid MBPP records found in {path} with keys prompt='{prompt_key}', code='{code_key}', test='{test_key}'"
        )
    return records


def synthetic_reference_stub(function_name: str) -> str:
    return textwrap.dedent(
        f"""
        def {function_name}(*args, **kwargs):
            raise NotImplementedError(\"Synthetic skeleton; generate solution before verify\")
        """
    ).strip()

def normalize_llm_synthetic_task(raw: dict, idx: int) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    prompt = safe_strip_text(raw.get("prompt") or raw.get("task") or raw.get("instruction"))
    function_name = safe_strip_text(raw.get("function_name") or raw.get("name"))
    tests = raw.get("test_list") or raw.get("tests") or []
    challenge_tests = raw.get("challenge_test_list") or raw.get("challenge_tests") or []
    setup_code = safe_strip_text(raw.get("test_setup_code") or raw.get("setup_code") or "")

    if isinstance(tests, str):
        tests = [x.strip() for x in tests.split("\n") if x.strip()]
    if isinstance(challenge_tests, str):
        challenge_tests = [x.strip() for x in challenge_tests.split("\n") if x.strip()]

    if not isinstance(tests, list):
        tests = [str(tests)]
    if not isinstance(challenge_tests, list):
        challenge_tests = [str(challenge_tests)] if challenge_tests else []

    tests, n_assert = normalize_test_lines([safe_strip_text(t) for t in tests])
    challenge_tests, _ = normalize_test_lines([safe_strip_text(t) for t in challenge_tests])

    if not prompt or not tests:
        return None
    if n_assert < MIN_SYNTHETIC_ASSERTS:
        logger.info("Loại đề tự sinh #%s: chỉ có %s assert hợp lệ (cần >= %s)", idx, n_assert, MIN_SYNTHETIC_ASSERTS)
        return None

    if not function_name:
        detected = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt)
        function_name = detected[0] if detected else f"solve_{idx}"
    elif not any(re.search(rf"\b{re.escape(function_name)}\b", t) for t in tests + [setup_code]):
        logger.info("Loại đề tự sinh #%s: test không gọi hàm '%s'", idx, function_name)
        return None

    # task_id gồm số thứ tự + hash nội dung: số thứ tự reset mỗi lần chạy nên bản cũ ("synllm_3") trùng giữa các lần.
    digest = hashlib.sha1(re.sub(r"\s+", " ", prompt.lower()).encode("utf-8")).hexdigest()[:8]
    return {
        "task_id": f"synllm_{idx}_{digest}",
        "function_name": function_name,
        "prompt": prompt,
        "reference_solution": synthetic_reference_stub(function_name),
        "test_setup_code": setup_code,
        "test_list": tests,
        "challenge_test_list": challenge_tests,
        "source_file": "synthetic_llm",
        "raw": raw,
    }
