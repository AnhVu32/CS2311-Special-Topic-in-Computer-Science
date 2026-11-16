import ast
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from agent0.core.config import RuntimeConfig
from agent0.core.llm_client import chat_completion
from agent0.core.utils import extract_code_block, strip_think_blocks
from agent0.execution.harness import run_harness
from agent0.execution.sandbox import record_tests, verify_solution

logger = logging.getLogger("agent0")

# Cách xử lý khi code đã sửa vẫn không qua test:
#   consensus (mặc định): sinh lời giải độc lập; nếu một lời giải qua hết test -> dùng nó; nếu các lời giải
#                         độc lập và lời giải hiện tại cùng cho một kết quả khác "expected" -> sửa expected
#                         theo kết quả đồng thuận; còn lại -> loại đề.
#   llm:  cách cũ — LLM phân xử CODE/TEST rồi LLM viết lại assert cho khớp code (dễ hợp thức hoá code sai).
#   none: không bao giờ sửa test; LLM phân xử, nếu CODE thì sửa code thêm một vòng.
TEST_REPAIR_MODES = ("consensus", "llm", "none")

SOLVER_SYSTEM_PROMPT = "You are a Python coding assistant. Return only runnable Python code, no markdown and no explanation."


@dataclass
class ExecutionResult:
    solution_code: str
    verification: Dict[str, Any]
    accepted: bool
    record: dict
    repair_rounds_used: int = 0
    test_repair_rounds_used: int = 0
    diagnosis_log: List[str] = field(default_factory=list)
    judge_verdict: Optional[str] = None
    reasoning: str = ""
    test_repair_method: Optional[str] = None
    consensus: Optional[Dict[str, Any]] = None
    solution_source: str = "model"


def _extract_solution(generated: str) -> str:
    text = strip_think_blocks(generated or "")
    return (extract_code_block(text) or text).strip()


def generate_solution(runtime: RuntimeConfig, record: dict) -> str:
    """Sinh lời giải bằng LLM. Lỗi gọi LLM được ném ra (không thay bằng lời giải mẫu của bộ dữ liệu)."""
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Implement Python code to satisfy this task and tests.

                Task:
                {record.get('prompt', '')}

                Tests:
                {json.dumps(record.get('test_list', []), ensure_ascii=False, indent=2)}

                Optional challenge tests:
                {json.dumps(record.get('challenge_test_list', []), ensure_ascii=False, indent=2)}

                Return only the Python solution code.
                """
            ).strip(),
        },
    ]
    generated = chat_completion(runtime, messages, temperature=0.1, max_tokens=1800)
    solution = _extract_solution(generated)
    logger.info("Generate solution done | task_id=%s chars=%s", record.get("task_id"), len(solution))
    return solution


def generate_reasoning(runtime: RuntimeConfig, record: dict, solution_code: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python coding tutor. Explain the approach and reasoning "
                "behind a solution in 3-5 concise bullet points. Focus on: "
                "1) What algorithm/technique is used "
                "2) Key steps of the approach "
                "3) Edge cases handled. "
                "No code. No markdown. Just plain text bullets."
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Task: {record.get('prompt', '')}

                Solution:
                {solution_code}

                Explain the reasoning behind this solution.
                """
            ).strip(),
        },
    ]
    try:
        reasoning = chat_completion(runtime, messages, temperature=0.1, max_tokens=500)
        reasoning = strip_think_blocks(reasoning)
        logger.info("Reasoning generated | task_id=%s chars=%s", record.get("task_id"), len(reasoning))
        return reasoning
    except Exception as e:
        logger.warning("Reasoning generation failed | task_id=%s error=%s", record.get("task_id"), e)
        return ""


def _clip_head(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[:n] + "\n...[truncated]"


def _clip_tail(text: str, n: int) -> str:
    """Giữ phần CUỐI (traceback/assert sai nằm ở cuối), không cắt mất chỗ có lỗi như bản cũ."""
    text = text or ""
    return text if len(text) <= n else "...[truncated]\n" + text[-n:]


def diagnose_failure(runtime: RuntimeConfig, record: dict, code: str, failure_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior Python debugger. Analyze the failure and explain the root cause. "
                "Be concise (3-5 sentences). Focus on: "
                "1) Is it a syntax error, logic error, or wrong test expectation? "
                "2) Which specific line or expression is wrong? "
                "3) What is the correct fix?"
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Task: {_clip_head(record.get('prompt', ''), 1200)}
                Tests: {_clip_head(json.dumps(record_tests(record), ensure_ascii=False), 2000)}
                Code:
                {_clip_head(code, 4000)}
                Error:
                {_clip_tail(failure_text, 1500)}

                Analyze: what exactly went wrong and where?
                """
            ).strip(),
        },
    ]
    try:
        diagnosis = chat_completion(runtime, messages, temperature=0.1, max_tokens=500)
        diagnosis = strip_think_blocks(diagnosis)
        logger.info("Diagnosis | task_id=%s\n%s", record.get("task_id"), diagnosis[:500])
        return diagnosis
    except Exception as e:
        logger.warning("Diagnosis failed | task_id=%s error=%s", record.get("task_id"), e)
        return ""


def repair_code(
    runtime: RuntimeConfig, record: dict, code: str, verification: Dict[str, Any],
    max_rounds: int = 2, timeout: int = 30,
) -> Tuple[str, Dict[str, Any], int, List[str]]:
    result = verification
    diagnoses: List[str] = []
    rounds = 0

    for r in range(max_rounds):
        if result.get("passed"):
            break

        failure_text = result.get("stderr") or result.get("stdout") or "Unknown failure"
        diagnosis = diagnose_failure(runtime, record, code, failure_text)
        diagnoses.append(diagnosis)

        messages = [
            {"role": "system", "content": "You are a senior Python engineer. Return only corrected Python code, without markdown fences or explanation."},
            {"role": "user", "content": textwrap.dedent(
                f"""
                Task: {record.get('prompt', '')}
                Tests: {json.dumps(record.get('test_list', []), ensure_ascii=False, indent=2)}
                Challenge tests: {json.dumps(record.get('challenge_test_list', []), ensure_ascii=False, indent=2)}
                Current code:
                {code}
                Error: {_clip_tail(failure_text, 3000)}
                Root cause: {diagnosis}

                Based on the analysis above, fix the code so it passes the tests.
                """
            ).strip()},
        ]
        try:
            logger.info("Repair code attempt %s | task_id=%s", r + 1, record.get("task_id"))
            repaired = chat_completion(runtime, messages, temperature=0.1, max_tokens=1800)
            extracted = _extract_solution(repaired)
            if extracted:
                code = extracted
                result = verify_solution(record, code, timeout=timeout)
                rounds += 1
                logger.info("Repair code result | task_id=%s round=%s passed=%s", record.get("task_id"), r + 1, result.get("passed"))
        except Exception as e:
            logger.warning("Repair code exception | task_id=%s round=%s error=%s", record.get("task_id"), r + 1, e)
            result = {"passed": False, "returncode": -1, "stdout": "", "stderr": f"Repair failed: {e}"}
            rounds += 1

    return code, result, rounds, diagnoses


def judge_failure(runtime: RuntimeConfig, record: dict, code: str, verification: Dict[str, Any]) -> str:
    """LLM phân xử CODE hay TEST sai (chỉ dùng ở chế độ test_repair llm / none).

    Bản cũ giới hạn 10 token nên model "thinking" luôn bị cắt trước khi kịp trả lời -> luôn ra CODE.
    Giờ cho đủ token và lấy từ CODE/TEST xuất hiện cuối cùng sau khi bỏ phần <think>.
    """
    failure_text = verification.get("stderr") or verification.get("stdout") or ""
    messages = [
        {"role": "system", "content": "You judge whether a test failure is caused by wrong CODE or wrong TEST assertions. Answer with exactly one word: CODE or TEST. Nothing else."},
        {"role": "user", "content": textwrap.dedent(
            f"""
            Task: {_clip_head(record.get('prompt', ''), 1200)}
            Code:
            {_clip_head(code, 4000)}
            Tests: {_clip_head(json.dumps(record_tests(record), ensure_ascii=False), 2000)}
            Error: {_clip_tail(failure_text, 1500)}

            Is the bug in the CODE or in the TEST assertions?
            """
        ).strip()},
    ]
    try:
        verdict = chat_completion(runtime, messages, temperature=0.0, max_tokens=400)
        words = re.findall(r"\b(CODE|TEST)\b", strip_think_blocks(verdict).upper())
        result = words[-1] if words else "CODE"
        logger.info("Judge verdict | task_id=%s → %s", record.get("task_id"), result)
        return result
    except Exception as e:
        logger.warning("Judge failed | task_id=%s error=%s", record.get("task_id"), e)
        return "CODE"


def repair_tests(
    runtime: RuntimeConfig, record: dict, code: str, verification: Dict[str, Any],
    max_rounds: int = 2, timeout: int = 30,
) -> Tuple[dict, Dict[str, Any], int, List[str]]:
    """Cách cũ (chế độ test_repair=llm): LLM viết lại assert cho khớp output của code."""
    patched = dict(record)
    result = verification
    diagnoses: List[str] = []
    rounds = 0

    for r in range(max_rounds):
        if result.get("passed"):
            break

        failure_text = result.get("stderr") or result.get("stdout") or "Unknown failure"
        diagnosis = diagnose_failure(runtime, record, code, failure_text)
        diagnoses.append(diagnosis)

        messages = [
            {"role": "system", "content": (
                "You are a senior Python test engineer. "
                "The solution code is correct, but the test assertions may be wrong. "
                "Return ONLY a JSON object with keys: test_list, challenge_test_list. "
                "Each is an array of corrected Python assert strings. No markdown. No explanation."
            )},
            {"role": "user", "content": textwrap.dedent(
                f"""
                Task: {record.get('prompt', '')}
                Solution code (correct):
                {code}
                Current test_list: {json.dumps(patched.get('test_list', []), ensure_ascii=False, indent=2)}
                Current challenge_test_list: {json.dumps(patched.get('challenge_test_list', []), ensure_ascii=False, indent=2)}
                Error: {_clip_tail(failure_text, 3000)}
                Root cause: {diagnosis}

                Fix the assert statements to match the actual output. Return JSON only.
                """
            ).strip()},
        ]
        try:
            logger.info("Repair tests attempt %s | task_id=%s", r + 1, record.get("task_id"))
            raw = chat_completion(runtime, messages, temperature=0.1, max_tokens=1500)
            raw = strip_think_blocks(raw)
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)

            new_tests = parsed.get("test_list", [])
            new_challenge = parsed.get("challenge_test_list", [])
            if new_tests and isinstance(new_tests, list):
                patched["test_list"] = [t.strip() for t in new_tests if isinstance(t, str) and t.strip()]
            if isinstance(new_challenge, list):
                patched["challenge_test_list"] = [t.strip() for t in new_challenge if isinstance(t, str) and t.strip()]

            result = verify_solution(patched, code, timeout=timeout)
            rounds += 1
            logger.info("Repair tests result | task_id=%s round=%s passed=%s", record.get("task_id"), r + 1, result.get("passed"))
        except Exception as e:
            logger.warning("Repair tests exception | task_id=%s round=%s error=%s", record.get("task_id"), r + 1, e)
            result = {"passed": False, "returncode": -1, "stdout": "", "stderr": f"Test repair failed: {e}"}
            rounds += 1

    return patched, result, rounds, diagnoses


# ───────────────────────── sửa test theo đồng thuận ─────────────────────────

def _signature_hints(code: str, tests: List[str]) -> str:
    """Dòng khai báo (def/class) của các hàm/lớp mà test gọi tới — không lộ phần thân của lời giải hiện tại."""
    try:
        mod = ast.parse(code)
    except SyntaxError:
        return ""
    joined = "\n".join(tests)
    lines = []
    for node in mod.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and re.search(rf"\b{re.escape(node.name)}\b", joined):
            ret = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
            lines.append(f"def {node.name}({ast.unparse(node.args)}){ret}: ...")
        elif isinstance(node, ast.ClassDef) and re.search(rf"\b{re.escape(node.name)}\b", joined):
            lines.append(f"class {node.name}:")
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    lines.append(f"    def {item.name}({ast.unparse(item.args)}): ...")
    return "\n".join(lines)


def generate_independent_solution(
    runtime: RuntimeConfig, record: dict, signatures: str, examples: List[str], seed: int, temperature: float = 0.7,
    few_shot: str = "",
) -> str:
    """Lời giải viết lại từ đầu (không thấy code hiện tại, không thấy các assert đang tranh chấp).

    few_shot: các ví dụ đã giải (vd truy hồi từ KB) đặt trước đề — dùng khi đo p_hat của "model + KB"."""
    example_text = "\n".join(examples) if examples else "(none)"
    # Giữ nguyên thụt lề 12 dấu cách của bản cũ: với đề nhiều dòng dedent không cắt được gì, nên đổi thụt lề sẽ đổi
    # prompt (và khoá cache LLM) so với các lần chạy trước.
    body = textwrap.dedent(
            f"""
            Implement the following task from scratch.

            Task:
            {record.get('prompt', '')}

            Required signature(s):
            {signatures or '(use the function name used in the examples)'}

            These example assertions must hold:
            {example_text}

            Return only the Python solution code.
            """
        ).strip()
    if few_shot:
        body = f"Here are some solved example tasks for reference:\n\n{few_shot}\n\n{body}"
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": body},
    ]
    return _extract_solution(chat_completion(runtime, messages, temperature=temperature, max_tokens=1800, seed=seed))


def _is_literal(text: str) -> bool:
    try:
        ast.literal_eval(text)
        return True
    except Exception:
        return False


def consensus_repair(
    runtime: RuntimeConfig,
    record: dict,
    code: str,
    verification: Dict[str, Any],
    n_independent: int = 2,
    timeout: int = 30,
) -> Dict[str, Any]:
    """Quyết định code sai hay test sai bằng bằng chứng thực thi thay vì để LLM tự phán.

    1. Sinh tối đa n_independent lời giải độc lập (không thấy code hiện tại, không thấy các assert đang sai).
       Lời giải nào qua hết test gốc -> code cũ sai, dùng lời giải đó.
    2. Một lời giải độc lập chỉ được bỏ phiếu nếu nó qua mọi assert mà code hiện tại qua.
    3. Với từng assert sai dạng `assert L == R`: nếu code hiện tại và mọi lời giải độc lập hợp lệ cho cùng
       một giá trị L (là literal Python) -> coi R sai, thay bằng giá trị đồng thuận. Cần >= 2 phiếu.
    4. Không đạt -> 'unresolved' (đề bị loại).
    """
    tests = record_tests(record)
    setup = record.get("test_setup_code", "") or ""
    info: Dict[str, Any] = {"method": "consensus", "outcome": "unresolved", "independent": 0, "valid_voters": 1, "rewritten": []}

    def give_up(reason: str) -> Dict[str, Any]:
        info["reason"] = reason
        return {**info, "solution_code": code, "record": record, "verification": verification}

    base = run_harness(code, tests, setup, timeout=timeout)
    failures = list(base.get("failures") or [])
    if not base.get("loaded") or base.get("statement_errors") or not failures:
        return give_up("lời giải không nạp được / lỗi dòng lệnh trong test / không có assert sai cụ thể")
    for f in failures:
        if "actual" not in f:
            return give_up(f"assert #{f['index']} không ở dạng `assert L == R` hoặc code hiện tại lỗi khi tính L")

    failing_idx = {f["index"] for f in failures}
    passing_examples = [t for i, t in enumerate(tests) if i not in failing_idx][:6]
    signatures = _signature_hints(code, tests)
    voters: List[Dict[str, Any]] = [base.get("values") or {}]

    for k in range(max(0, n_independent)):
        try:
            candidate = generate_independent_solution(runtime, record, signatures, passing_examples, seed=1000 + k)
        except Exception as e:
            logger.warning("Independent solution failed | task_id=%s k=%s error=%s", record.get("task_id"), k, e)
            continue
        info["independent"] += 1
        res = run_harness(candidate, tests, setup, timeout=timeout)
        if res.get("passed"):
            info["outcome"] = "code_fixed"
            logger.info("Consensus: lời giải độc lập #%s qua hết test gốc | task_id=%s", k + 1, record.get("task_id"))
            return {**info, "solution_code": candidate, "record": record,
                    "verification": verify_solution(record, candidate, timeout=timeout)}
        cand_fail = {f["index"] for f in res.get("failures") or []}
        if res.get("loaded") and not res.get("statement_errors") and cand_fail.issubset(failing_idx):
            voters.append(res.get("values") or {})

    info["valid_voters"] = len(voters)
    if len(voters) < 2:
        return give_up("không có lời giải độc lập hợp lệ để đối chiếu")

    new_tests = list(tests)
    for f in failures:
        idx = f["index"]
        votes = []
        for values in voters:
            v = values.get(str(idx)) or {}
            votes.append(v.get("repr") if v.get("ok") else None)
        if any(v is None for v in votes) or len(set(votes)) != 1 or not _is_literal(votes[0]):
            return give_up(f"các lời giải không đồng thuận ở assert #{idx}")
        new_tests[idx] = f"assert {f['left']} == {votes[0]}"
        info["rewritten"].append({"index": idx, "old": tests[idx], "new": new_tests[idx]})

    n_main = len(record.get("test_list", []) or [])
    patched = dict(record)
    patched["test_list"] = new_tests[:n_main]
    patched["challenge_test_list"] = new_tests[n_main:]
    result = verify_solution(patched, code, timeout=timeout)
    if not result["passed"]:
        return give_up("sau khi sửa theo đồng thuận vẫn chưa qua")
    info["outcome"] = "tests_fixed"
    logger.info("Consensus: sửa %s assert theo đồng thuận %s phiếu | task_id=%s",
                len(info["rewritten"]), info["valid_voters"], record.get("task_id"))
    return {**info, "solution_code": code, "record": patched, "verification": result}


# ───────────────────────── luồng chính ─────────────────────────

def _finalize(
    runtime: RuntimeConfig,
    record: dict,
    solution_code: str,
    verification: Dict[str, Any],
    repair_rounds: int,
    verify_timeout: int,
    test_repair_mode: str,
    consensus_solutions: int,
    with_reasoning: bool,
) -> ExecutionResult:
    if test_repair_mode not in TEST_REPAIR_MODES:
        raise ValueError(f"test_repair_mode không hợp lệ: {test_repair_mode}")
    task_id = record.get("task_id", "?")
    all_diagnoses: List[str] = []
    repair_rounds_used = 0
    test_repair_rounds_used = 0
    judge_verdict = None
    patched_record = record
    method = None
    consensus = None
    source = "model"

    def done(code: str, ver: Dict[str, Any], rec: dict) -> ExecutionResult:
        reasoning = generate_reasoning(runtime, rec, code) if (with_reasoning and ver.get("passed")) else ""
        return ExecutionResult(
            solution_code=code, verification=ver, accepted=bool(ver.get("passed")), record=rec,
            repair_rounds_used=repair_rounds_used, test_repair_rounds_used=test_repair_rounds_used,
            diagnosis_log=all_diagnoses, judge_verdict=judge_verdict, reasoning=reasoning,
            test_repair_method=method, consensus=consensus, solution_source=source,
        )

    if verification["passed"] or repair_rounds <= 0 or runtime.provider == "offline":
        return done(solution_code, verification, record)

    solution_code, verification, repair_rounds_used, diagnoses = repair_code(
        runtime, record, solution_code, verification, max_rounds=repair_rounds, timeout=verify_timeout,
    )
    all_diagnoses.extend(diagnoses)
    if verification["passed"]:
        return done(solution_code, verification, record)

    if test_repair_mode == "consensus":
        method = "consensus"
        outcome = consensus_repair(runtime, record, solution_code, verification, consensus_solutions, verify_timeout)
        consensus = {k: outcome[k] for k in ("outcome", "independent", "valid_voters", "rewritten") if k in outcome}
        if "reason" in outcome:
            consensus["reason"] = outcome["reason"]
        judge_verdict = {"code_fixed": "CODE", "tests_fixed": "TEST"}.get(outcome["outcome"], "UNRESOLVED")
        if outcome["outcome"] == "code_fixed":
            source = "independent"
        if outcome["outcome"] == "tests_fixed":
            test_repair_rounds_used = 1
            patched_record = outcome["record"]
        return done(outcome["solution_code"], outcome["verification"], patched_record)

    judge_verdict = judge_failure(runtime, record, solution_code, verification)
    if judge_verdict == "TEST" and test_repair_mode == "llm":
        method = "llm"
        patched_record, verification, test_repair_rounds_used, diagnoses = repair_tests(
            runtime, record, solution_code, verification, max_rounds=repair_rounds, timeout=verify_timeout,
        )
        all_diagnoses.extend(diagnoses)
    elif judge_verdict == "CODE":
        logger.info("Judge says CODE wrong, one more repair | task_id=%s", task_id)
        solution_code, verification, extra, diagnoses = repair_code(
            runtime, record, solution_code, verification, max_rounds=1, timeout=verify_timeout,
        )
        repair_rounds_used += extra
        all_diagnoses.extend(diagnoses)
    return done(solution_code, verification, patched_record)


def execute_task(
    runtime: RuntimeConfig,
    record: dict,
    repair_rounds: int = 2,
    verify_timeout: int = 20,
    test_repair_mode: str = "consensus",
    consensus_solutions: int = 2,
    with_reasoning: bool = True,
) -> ExecutionResult:
    """Sinh lời giải -> kiểm chứng -> sửa code -> (đồng thuận / phân xử) -> reasoning cho mục được nhận.

    provider 'offline': không gọi LLM, dùng lời giải mẫu của bộ dữ liệu (solution_source = 'reference'),
    chỉ kiểm chứng — để chạy thử pipeline hoặc dựng KB "lời giải gốc" làm mốc so sánh.
    """
    task_id = record.get("task_id", "?")
    logger.info("Execute start | task_id=%s", task_id)
    if runtime.provider == "offline":
        reference = (record.get("reference_solution") or "").strip()
        if not reference or "Synthetic skeleton" in reference:
            raise RuntimeError("provider offline cần lời giải mẫu trong dữ liệu (đề tự sinh không có)")
        verification = verify_solution(record, reference, timeout=verify_timeout)
        return ExecutionResult(
            solution_code=reference, verification=verification, accepted=bool(verification.get("passed")),
            record=record, solution_source="reference",
        )
    solution_code = generate_solution(runtime, record)
    verification = verify_solution(record, solution_code, timeout=verify_timeout)
    return _finalize(
        runtime, record, solution_code, verification, repair_rounds, verify_timeout,
        test_repair_mode, consensus_solutions, with_reasoning=with_reasoning,
    )


def solve_with_knowledge(
    runtime: RuntimeConfig,
    record: dict,
    few_shot_text: str,
    repair_rounds: int = 2,
    verify_timeout: int = 20,
    test_repair_mode: str = "consensus",
    consensus_solutions: int = 2,
) -> ExecutionResult:
    task_id = record.get("task_id", "?")
    logger.info("Solve with KB | task_id=%s", task_id)
    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Here are similar solved tasks for reference:

                {few_shot_text}

                Now solve the new task below following the same patterns.

                Task:
                {record.get('prompt', '')}

                Tests:
                {json.dumps(record.get('test_list', []), ensure_ascii=False, indent=2)}

                Optional challenge tests:
                {json.dumps(record.get('challenge_test_list', []), ensure_ascii=False, indent=2)}

                Return only the Python solution code.
                """
            ).strip(),
        },
    ]
    try:
        solution_code = _extract_solution(chat_completion(runtime, messages, temperature=0.1, max_tokens=1800))
        logger.info("KB-solve generated | task_id=%s chars=%s", task_id, len(solution_code))
    except Exception as e:
        logger.warning("KB-solve generation failed | task_id=%s error=%s", task_id, e)
        return ExecutionResult(
            solution_code="", verification={"passed": False, "stderr": str(e)}, accepted=False, record=record,
        )
    verification = verify_solution(record, solution_code, timeout=verify_timeout)
    return _finalize(
        runtime, record, solution_code, verification, repair_rounds, verify_timeout,
        test_repair_mode, consensus_solutions, with_reasoning=False,
    )
