#!/usr/bin/env python3
"""Smoke test không cần mạng, không cần API key, không cần evalplus/gradio.

Chạy từ thư mục Source_Code:
    python tests/smoke_test.py

Kiểm tra: chọn provider/nạp .env, retry khi gọi LLM, cache LLM, sinh lời giải không tạo stub + --resume
(+ chặn resume khác cấu hình), song song + ví dụ few-shot, kiểm tra file samples, đọc kết quả EvalPlus, McNemar,
giao thức prompt, tách code, harness chặt, chuẩn hoá test tự sinh, đồng thuận sửa test, p_hat, planner, taxonomy,
khử ô nhiễm, retriever (index theo hash, lọc, ngưỡng, MMR, định dạng), pipeline run_curriculum (tự sinh + offline),
tập dev, và `--help` của các script. LLM, embedding và EvalPlus được thay bằng bản giả.
"""
import json
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

SOURCE_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOURCE_CODE))

import agent0.core.config as config  # noqa: E402
import agent0.core.llm_client as llm_client  # noqa: E402
import agent0.scripts.run_benchmark as rb  # noqa: E402
from agent0.core.config import RuntimeConfig, resolve_runtime  # noqa: E402
from agent0.core.utils import extract_code_block  # noqa: E402
from agent0.execution.strict_check import run_tests_isolated  # noqa: E402

KEYS = ("GROQ_API_KEY", "OLLAMA_API_KEY", "OPENAI_API_KEY")
FAKE_TASKS = [
    {"task_id": f"Mbpp/{i}", "prompt": f'"""\nWrite a function f{i}.\nassert f{i}(1) == 1\n"""\n',
     "entry_point": f"f{i}", "test_list": [f"assert f{i}(1) == 1", f"assert f{i}(2) == 2"]}
    for i in (2, 3, 4)
]
FAKE_IDS = [t["task_id"] for t in FAKE_TASKS]


class EnvPatch:
    """Đặt tạm biến môi trường (và tắt việc đọc .env thật) trong khối with."""

    def __init__(self, **values):
        self.values = values

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in set(KEYS) | set(self.values)}
        self.saved_loaded = config._ENV_LOADED
        config._ENV_LOADED = True
        for k in KEYS:
            os.environ.pop(k, None)
        for k, v in self.values.items():
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        config._ENV_LOADED = self.saved_loaded


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"Không thấy {exc_type.__name__}")


def fake_llm(fail_ids=()):
    """chat_completion giả: trả code đúng cho bài, hoặc raise LLMCallError cho các bài trong fail_ids."""
    calls = []

    def _fake(runtime, messages, **kwargs):
        text = messages[-1]["content"]
        calls.append(kwargs)
        for t in FAKE_TASKS:
            if f"Write a function {t['entry_point']}." in text:
                if t["task_id"] in fail_ids:
                    raise llm_client.LLMCallError("giả lập: hết lượt thử")
                return f"Đây là lời giải:\n```python\ndef {t['entry_point']}(x):\n    return x\n```"
        return "```python\ndef unknown():\n    return None\n```"

    _fake.calls = calls
    return _fake


# ───────────────────────────── tests ─────────────────────────────

def test_resolve_runtime():
    with EnvPatch():
        rt = resolve_runtime("auto", "gemma3:12b")
        assert rt.provider == "ollama" and rt.base_url == config.OLLAMA_BASE_URL, rt
        assert resolve_runtime("offline", "x").provider == "offline"
        err = expect_raises(ValueError, resolve_runtime, "groq", "llama-3.3-70b-versatile")
        assert "GROQ_API_KEY" in str(err)
        expect_raises(ValueError, resolve_runtime, "custom", "m")
        expect_raises(ValueError, resolve_runtime, "nope", "m")
    with EnvPatch(OLLAMA_API_KEY="k1"):
        rt = resolve_runtime("auto", "gemma3:12b")
        assert rt.provider == "ollama-cloud" and rt.base_url == config.OLLAMA_CLOUD_BASE_URL and rt.api_key == "k1", rt
    with EnvPatch(GROQ_API_KEY="k2"):
        rt = resolve_runtime("auto", "llama-3.3-70b-versatile")
        assert rt.provider == "groq" and rt.base_url == config.GROQ_BASE_URL and rt.api_key == "k2", rt
        rt = resolve_runtime("groq", "llama-3.1-8b-instant", api_key="cli")
        assert rt.api_key == "cli"
        info = config.describe_runtime(rt)
        assert "api_key" not in info and info["api_key_set"] is True


def test_env_file_fallback_parser():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / ".env"
        p.write_text('# comment\nexport A_KEY="x y"\nB_KEY=plain # note\nC_KEY=\'q\'\nBROKEN\n', encoding="utf-8")
        values = config._parse_env_file(p)
        assert values == {"A_KEY": "x y", "B_KEY": "plain", "C_KEY": "q"}, values


def test_llm_retry_and_seed():
    rt = RuntimeConfig(provider="ollama", model="m", backend="b", base_url="http://localhost:11434")
    calls, sleeps = [], []
    orig_post, orig_sleep = llm_client.http_post_json, llm_client.time.sleep
    try:
        llm_client.time.sleep = lambda s: sleeps.append(s)

        def flaky(url, payload, timeout=300, headers=None):
            calls.append(payload)
            if len(calls) <= 2:
                raise llm_client.HTTPStatusError(429, url, "rate limit", retry_after=1.0)
            return {"message": {"content": "ok"}}

        llm_client.http_post_json = flaky
        assert llm_client.chat_completion(rt, [{"role": "user", "content": "hi"}], seed=7, max_retries=3) == "ok"
        assert len(calls) == 3 and len(sleeps) == 2 and calls[0]["options"]["seed"] == 7

        calls.clear()

        def unauthorized(url, payload, timeout=300, headers=None):
            calls.append(1)
            raise llm_client.HTTPStatusError(401, url, "bad key")

        llm_client.http_post_json = unauthorized
        expect_raises(llm_client.LLMCallError, llm_client.chat_completion, rt, [{"role": "user", "content": "x"}])
        assert len(calls) == 1, "401 không được thử lại"

        calls.clear()

        def always_503(url, payload, timeout=300, headers=None):
            calls.append(1)
            raise llm_client.HTTPStatusError(503, url, "down")

        llm_client.http_post_json = always_503
        expect_raises(llm_client.LLMCallError, llm_client.chat_completion, rt, [{"role": "user", "content": "x"}], max_retries=2)
        assert len(calls) == 3
    finally:
        llm_client.http_post_json, llm_client.time.sleep = orig_post, orig_sleep


def test_generate_no_stub_and_resume():
    rt = RuntimeConfig(provider="ollama", model="m", backend="b", base_url="http://x")
    orig = rb.chat_completion
    try:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "m" / "baseline_samples.jsonl"
            rb.chat_completion = fake_llm(fail_ids={"Mbpp/3"})
            stats = rb.generate_all(rt, FAKE_TASKS, out)
            rows = rb.read_jsonl(out)
            assert [r["task_id"] for r in rows] == ["Mbpp/2", "Mbpp/4"], rows
            assert stats["errors"] == 1 and not any(rb.is_stub(r["solution"]) for r in rows)
            errors = rb.read_jsonl(rb.sidecar(out, "errors.jsonl"))
            assert errors[0]["task_id"] == "Mbpp/3" and "LLMCallError" in errors[0]["error"]

            expect_raises(FileExistsError, rb.generate_all, rt, FAKE_TASKS, out)

            # Lần chạy cũ có stub -> --resume bỏ stub và sinh lại; không tạo dòng trùng.
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"task_id": "Mbpp/3", "solution": "def f3(*args, **kwargs):\n    pass"}) + "\n")
            rb.chat_completion = fake_llm()
            stats = rb.generate_all(rt, FAKE_TASKS, out, resume=True)
            rows = rb.read_jsonl(out)
            assert sorted(r["task_id"] for r in rows) == FAKE_IDS and stats["generated"] == 1 and stats["reused"] == 2
            rep = rb.check_samples(out, FAKE_IDS)
            assert rep["complete"], rep
    finally:
        rb.chat_completion = orig


def test_check_samples_flags():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x_samples.jsonl"
        rows = [
            {"task_id": "Mbpp/2", "solution": "def f2(x):\n    return x"},
            {"task_id": "Mbpp/2", "solution": "def f2(x):\n    return x"},
            {"task_id": "Mbpp/3", "solution": "def f3(*args, **kwargs):\n    pass"},
            {"task_id": "Mbpp/999", "solution": "def g():\n    pass"},
        ]
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        rep = rb.check_samples(p, FAKE_IDS)
        assert rep["missing"] == ["Mbpp/4"] and rep["stubs"] == ["Mbpp/3"]
        assert rep["duplicates"] == ["Mbpp/2"] and rep["unknown"] == ["Mbpp/999"] and not rep["complete"]


def test_evaluate_flow_with_fake_evalplus():
    """evaluate: từ chối file thiếu bài; --allow_incomplete tạo bản đệm; metrics đọc từ *_eval_results.json."""
    orig_run, orig_ids = rb.run_evalplus, rb.load_expected_task_ids
    seen = []

    def fake_run_evalplus(eval_file, evaluator, force, parallel=None):
        seen.append((Path(eval_file).name, evaluator, force))
        solutions = {r["task_id"]: r["solution"] for r in rb.read_jsonl(eval_file)}
        result = {"eval": {}}
        for tid, sol in solutions.items():
            base = "pass" if "return x" in sol else "fail"
            plus = "pass" if (base == "pass" and tid != "Mbpp/4") else "fail"
            result["eval"][tid] = [{"base_status": base, "plus_status": plus}]
        rb.sidecar(eval_file, "eval_results.json").write_text(json.dumps(result), encoding="utf-8")

    try:
        rb.run_evalplus = fake_run_evalplus
        rb.load_expected_task_ids = lambda path=None: list(FAKE_IDS)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "with_kb_n2_samples.jsonl"
            p.write_text("".join(json.dumps({"task_id": t, "solution": f"def f(x):\n    return x"}) + "\n"
                                 for t in FAKE_IDS[:2]), encoding="utf-8")
            expect_raises(rb.IncompleteSamplesError, rb.evaluate_samples, p, "local")
            m = rb.evaluate_samples(p, "local", allow_incomplete=True)
            assert m["partial"] and m["n_missing"] == 1 and seen[-1][0] == "with_kb_n2_samples_padded.jsonl"
            assert m["base_pass@1"] == round(100 * 2 / 3, 2) and m["attempted_only"]["base_pass@1"] == 100.0

            with p.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"task_id": "Mbpp/4", "solution": "def f4(x):\n    return x"}) + "\n")
            m = rb.evaluate_samples(p, "local")
            assert not m["partial"] and m["base_pass@1"] == 100.0 and m["plus_pass@1"] == round(100 * 2 / 3, 2), m
            assert json.loads(rb.sidecar(p, "metrics.json").read_text(encoding="utf-8"))["plus_pass@1"] == m["plus_pass@1"]
            n_runs = len(seen)
            rb.evaluate_samples(p, "local")  # đã có kết quả mới hơn samples -> không chấm lại
            assert len(seen) == n_runs
    finally:
        rb.run_evalplus, rb.load_expected_task_ids = orig_run, orig_ids


def test_mcnemar_and_compare():
    assert abs(rb.mcnemar_exact(14, 12) - 0.8450) < 1e-3
    assert rb.mcnemar_exact(0, 0) == 1.0 and rb.mcnemar_exact(10, 0) < 0.01
    va = {"a": (1.0, 1.0), "b": (0.0, 0.0), "c": (1.0, 0.0)}
    vb = {"a": (1.0, 0.0), "b": (1.0, 1.0), "c": (1.0, 1.0)}
    cmp = rb.compare_verdicts(va, vb, ["a", "b", "c"])
    assert cmp["base"]["a_fail_b_pass"] == 1 and cmp["base"]["a_pass_b_fail"] == 0
    assert cmp["plus"]["a_fail_b_pass"] == 2 and cmp["plus"]["a_pass_b_fail"] == 1


def test_prompt_styles():
    task = dict(FAKE_TASKS[0])
    msgs = rb.build_messages(task, "", "evalplus")
    assert rb.EVALPLUS_INSTRUCTION in msgs[-1]["content"] and "Tests:" not in msgs[-1]["content"]
    assert "assert f2(2) == 2" not in msgs[-1]["content"], "prompt chuẩn không được lộ base test"
    msgs = rb.build_messages(task, "EXAMPLES", "with_tests")
    assert "Tests:" in msgs[-1]["content"] and "assert f2(2) == 2" in msgs[-1]["content"] and "EXAMPLES" in msgs[-1]["content"]


def test_extract_code_block():
    assert extract_code_block("```py\ndef a():\n    return 1\n```") == "def a():\n    return 1"
    assert extract_code_block("text\n```\ndef b():\n    return 2\n```\n") == "def b():\n    return 2"
    think = "<think>```python\ndef wrong():\n    pass\n```</think>\n```python\ndef right():\n    return 3\n```"
    assert extract_code_block(think) == "def right():\n    return 3"
    assert extract_code_block("```python\ndef cut():\n    return 4") == "def cut():\n    return 4"
    assert extract_code_block("```bash\nls\n```") is None and extract_code_block("no code") is None


def test_strict_check():
    ok = run_tests_isolated("def f(x):\n    return x", ["assert f(1) == 1", "assert f(2) == 2"])
    assert ok["passed"] and ok["n_ok"] == 2, ok
    wrong_exit = run_tests_isolated("import sys\ndef f(x):\n    return 0\nsys.exit(0)", ["assert f(1) == 1"])
    assert not wrong_exit["passed"], wrong_exit
    demo_exit = run_tests_isolated(
        "import sys\ndef f(x):\n    return 0\nif __name__ == '__main__':\n    sys.exit(0)", ["assert f(1) == 1"]
    )
    assert not demo_exit["passed"] and demo_exit["n_ok"] == 0, demo_exit
    os.environ["AGENT0_SMOKE_SECRET"] = "s3cret"
    try:
        leak = run_tests_isolated("import os\nSECRET = os.environ.get('AGENT0_SMOKE_SECRET')", ["assert SECRET is None"])
        assert leak["passed"], "tiến trình con không được thấy biến môi trường của tiến trình cha"
    finally:
        os.environ.pop("AGENT0_SMOKE_SECRET", None)


def test_paths_and_dev_split():
    ids = rb.load_expected_task_ids()
    assert len(ids) == 378 and len(set(ids)) == 378
    assert config.DEFAULT_DATA_FILE.exists(), config.DEFAULT_DATA_FILE
    tasks = rb.load_dev_tasks(dev_size=None)
    assert len(tasks) == 596, len(tasks)
    assert not ({t["task_id"].split("/")[1] for t in tasks} & {i.split("/")[1] for i in ids})
    assert tasks[0]["prompt"].startswith('"""\n') and tasks[0]["test_list"]
    small = rb.load_dev_tasks(dev_size=50, dev_seed=1)
    assert len(small) == 50 and small == rb.load_dev_tasks(dev_size=50, dev_seed=1)
    if not config.DEFAULT_KB_PATH.exists():
        print(f"   (cảnh báo) chưa có KB mặc định {config.DEFAULT_KB_PATH}")


def test_cli_help():
    env = {**os.environ, "PYTHONPATH": str(SOURCE_CODE)}
    for mod in ("run_benchmark", "run_curriculum", "run_planner", "run_demo"):
        proc = subprocess.run([sys.executable, "-m", f"agent0.scripts.{mod}", "--help"], cwd=SOURCE_CODE,
                              capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0, f"{mod} --help lỗi:\n{proc.stderr[-1500:]}"
    for sub in ("generate", "check", "evaluate", "run", "compare", "dev"):
        proc = subprocess.run([sys.executable, "-m", "agent0.scripts.run_benchmark", sub, "--help"], cwd=SOURCE_CODE,
                              capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0 and "--" in proc.stdout, f"run_benchmark {sub} --help lỗi:\n{proc.stderr[-1500:]}"
    for tool in ("run_matrix", "summarize_matrix", "split_kb_by_phat", "package_for_kaggle", "clean_kb",
                 "hf_generate", "grpo_executor"):
        proc = subprocess.run([sys.executable, f"tools/{tool}.py", "--help"], cwd=SOURCE_CODE,
                              capture_output=True, text=True, env=env, timeout=120)
        assert proc.returncode == 0 and "--" in proc.stdout, f"tools/{tool}.py --help lỗi:\n{proc.stderr[-1500:]}"


# ───────────────────── Ưu tiên 2–6: harness, đồng thuận, p_hat, KB, retrieval, hiệu năng ─────────────────────

from agent0.execution.harness import format_report, run_harness  # noqa: E402
import agent0.execution.agent as agent_mod  # noqa: E402

RT_FAKE = RuntimeConfig(provider="ollama", model="fake", backend="fake", base_url="http://x")
ADD_GOOD = "def add(a, b):\n    return a + b"
ADD_BAD = "def add(a, b):\n    return a - b"
ADD_ALT = "def add(a, b):\n    return a * b"


def test_harness_details():
    r = run_harness(ADD_BAD, ["assert add(1, 2) == 3", "add(2, 2) == 4"])
    assert not r["passed"] and r["n_asserts"] == 2 and r["converted"] == 1, r
    f = r["failures"][0]
    assert f["actual"] == "-1" and f["expected"] == "3", f
    assert "returned -1, expected 3" in format_report(r), format_report(r)
    assert not run_harness("def g():\n    return 1", ["assert 1 == 1"])["passed"], "test không gọi hàm của lời giải -> không tính"
    assert not run_harness("import sys\ndef add(a, b):\n    return 0\nsys.exit(0)", ["assert add(1, 1) == 2"])["passed"]
    # setup định nghĩa class mà lời giải dùng (thứ tự setup -> solution), và setup có stub bị lời giải ghi đè
    assert run_harness("def area(s):\n    return s.w * s.h", ["assert area(Box(2, 3)) == 6"],
                       setup="class Box:\n    def __init__(self, w, h):\n        self.w, self.h = w, h")["passed"]
    assert run_harness(ADD_GOOD, ["assert add(2, 3) == 5"], setup="def add(a, b):\n    pass")["passed"]
    # test có trạng thái: vế trái chỉ được tính một lần
    stateful = "class C:\n    def __init__(self):\n        self.n = 0\n    def inc(self):\n        self.n += 1\n        return self.n"
    r = run_harness(stateful, ["c = C()", "assert c.inc() == 1", "assert c.inc() == 3"])
    assert not r["passed"] and r["failures"][0]["actual"] == "2", r["failures"]


def test_synthetic_task_normalization():
    from agent0.data.mbpp_loader import normalize_llm_synthetic_task, normalize_test_lines

    kept, n = normalize_test_lines(["add(1, 2) == 3", "{'input': 1, 'expected': 2}", "assert add(0, 0) == 0", "x = (", "s = 1"])
    assert kept == ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "s = 1"] and n == 2, kept
    raw = {"prompt": "Add two numbers `add(a, b)`.", "function_name": "add",
           "test_list": ["assert add(1, 2) == 3", "assert add(2, 2) == 4", "assert add(0, 5) == 5"]}
    a = normalize_llm_synthetic_task(raw, 1)
    b = normalize_llm_synthetic_task(dict(raw, prompt="Multiply two numbers `add(a, b)`."), 1)
    assert a and b and a["task_id"].startswith("synllm_1_") and a["task_id"] != b["task_id"], (a, b)
    assert normalize_llm_synthetic_task(dict(raw, test_list=raw["test_list"][:2]), 2) is None, "< 3 assert -> loại"
    assert normalize_llm_synthetic_task(dict(raw, function_name="sub"), 3) is None, "test không gọi function_name -> loại"


def _agent_fake(first, independents, judge="TEST"):
    queue = list(independents)

    def fake(runtime, messages, **kw):
        sysmsg, user = messages[0]["content"], messages[-1]["content"]
        if "judge" in sysmsg:
            return "<think>CODE or TEST?</think> " + judge
        if "tutor" in sysmsg:
            return "reasoning"
        if "from scratch" in user:
            return "```python\n" + (queue.pop(0) if queue else ADD_BAD) + "\n```"
        return "```python\n" + first + "\n```"

    return fake


def test_consensus_test_repair():
    ok_tests = {"task_id": "t1", "prompt": "Add two numbers", "challenge_test_list": [],
                "test_list": ["assert add(1, 2) == 3", "assert add(2, 2) == 4", "assert add(0, 5) == 5"]}
    bad_test = {"task_id": "t2", "prompt": "Add two numbers", "challenge_test_list": ["assert add(10, 1) == 11"],
                "test_list": ["assert add(1, 2) == 3", "assert add(2, 2) == 5", "assert add(0, 5) == 5"]}
    orig = agent_mod.chat_completion
    try:
        agent_mod.chat_completion = _agent_fake(ADD_BAD, [ADD_GOOD])
        r = agent_mod.execute_task(RT_FAKE, ok_tests, repair_rounds=1, verify_timeout=10)
        assert r.accepted and r.solution_source == "independent" and r.judge_verdict == "CODE", r

        agent_mod.chat_completion = _agent_fake(ADD_GOOD, [ADD_GOOD, ADD_GOOD])
        r = agent_mod.execute_task(RT_FAKE, bad_test, repair_rounds=1, verify_timeout=10)
        assert r.accepted and r.judge_verdict == "TEST" and r.record["test_list"][1] == "assert add(2, 2) == 4", r.record

        agent_mod.chat_completion = _agent_fake(ADD_GOOD, [ADD_ALT, ADD_ALT])
        r = agent_mod.execute_task(RT_FAKE, bad_test, repair_rounds=1, verify_timeout=10)
        assert not r.accepted and r.judge_verdict == "UNRESOLVED", r.consensus

        agent_mod.chat_completion = _agent_fake(ADD_GOOD, [])
        r = agent_mod.execute_task(RT_FAKE, bad_test, repair_rounds=1, verify_timeout=10, test_repair_mode="none")
        assert not r.accepted and r.test_repair_method is None

        agent_mod.chat_completion = lambda rt, m, **k: "<think>maybe TEST?</think>\nThe bug is in the CODE."
        assert agent_mod.judge_failure(RT_FAKE, ok_tests, ADD_BAD, {"stderr": "x"}) == "CODE"

        def boom(*a, **k):
            raise RuntimeError("api down")

        agent_mod.chat_completion = boom
        expect_raises(RuntimeError, agent_mod.generate_solution, RT_FAKE,
                      {"task_id": "m", "prompt": "p", "reference_solution": ADD_GOOD, "test_list": []})
        # offline: dùng lời giải mẫu của dữ liệu, không gọi LLM
        r = agent_mod.execute_task(RuntimeConfig(provider="offline", model="x", backend="offline"),
                                   dict(ok_tests, reference_solution=ADD_GOOD), verify_timeout=10)
        assert r.accepted and r.solution_source == "reference"
    finally:
        agent_mod.chat_completion = orig


def test_uncertainty_p_hat():
    from agent0.curriculum import uncertainty as unc

    rec = {"task_id": "u", "prompt": "Add two numbers", "challenge_test_list": [],
           "test_list": ["assert add(1, 2) == 3", "assert add(2, 2) == 4", "assert add(0, 5) == 5"]}
    seeds = []

    def fake(runtime, messages, seed=None, temperature=None, **kw):
        seeds.append((seed, temperature))
        return "```python\n" + (ADD_GOOD if seed % 2 == 0 else ADD_BAD) + "\n```"

    orig = agent_mod.chat_completion
    try:
        agent_mod.chat_completion = fake
        out = unc.estimate_pass_rate(RT_FAKE, rec, ADD_GOOD, k=4, temperature=0.8, timeout=10, base_seed=100)
    finally:
        agent_mod.chat_completion = orig
    assert out["p_hat"] == 0.5 and out["r_unc"] == 1.0 and out["k"] == 4, out
    assert [s for s, _ in seeds] == [100, 101, 102, 103] and all(t == 0.8 for _, t in seeds)
    assert unc.band_verdict(1.0) == "too_easy" and unc.band_verdict(0.0) == "too_hard" and unc.band_verdict(0.25) == "in_band"
    groups = unc.summarize_by_group([{"group": "a", "difficulty": "easy", "p_hat": 1.0}, {"group": "a", "difficulty": "easy", "p_hat": 0.5}])
    assert groups[0]["p_hat_mean"] == 0.75 and groups[0]["too_easy"] == 1


def test_planner_task_counts():
    from agent0.curriculum.planner import _adjust_task_counts, _feedback_instructions

    for n, per, target in [(20, 2, 30), (50, 2, 30), (5, 3, 30), (3, "2", 7)]:
        out = _adjust_task_counts([{"name": f"s{i}", "task_count": per} for i in range(n)], target)
        counts = [x["task_count"] for x in out]
        assert sum(counts) == target and min(counts) >= 1, (n, per, target, counts)
    lines = _feedback_instructions([{"subtopic": "Strings", "difficulty": "easy", "p_hat_mean": 1.0, "n": 3}])
    assert any("Strings" in x and "p_hat=1.0" in x for x in lines)


def test_taxonomy_rules():
    from agent0.curriculum.strategies import infer_taxonomy, order_all_strategies

    meta = infer_taxonomy({"prompt": "Write a function to find the sum of a list, for each element read once.",
                           "reference_solution": "def f(xs):\n    t = 0\n    for x in xs:\n        if x and x or x:\n            t += x\n    return t"})
    for wrong in ("io", "loops", "bitwise", "search"):
        assert wrong not in meta["signals"], meta["signals"]
    assert "lists" in meta["signals"]
    recs = [dict(infer_taxonomy({"prompt": p}), task_id=i, prompt=p) for i, p in enumerate(["sort a list", "reverse a string", "count words"])]
    sel = order_all_strategies(recs, 3)
    assert len(sel) == len({r["task_id"] for r in sel}) == 3, "mỗi bài chỉ được chọn một lần"


def test_decontamination_index():
    from agent0.data.mbpp_loader import load_mbpp_plus_ids
    from agent0.data.text_similarity import build_mbpp_plus_index, prompt_key

    assert prompt_key("  Write  a Function.\n") == "write a function"
    idx = build_mbpp_plus_index(config.DEFAULT_DATA_FILE, load_mbpp_plus_ids())
    assert len(idx) == 378
    sim, pid, _ = idx.most_similar("Write a function to find the area of a rectangle.")
    assert sim > 0.9 and pid == "Mbpp/458", (sim, pid)
    sim, _, _ = idx.most_similar("Implement Dijkstra shortest paths on a weighted adjacency matrix with a heap.")
    assert sim < 0.5, sim


def _fake_embed(texts, **kw):
    import hashlib as _h

    vecs = []
    for t in texts:
        v = [0.0] * 64
        for w in t.lower().replace(".", " ").split():
            v[int(_h.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
        vecs.append(v)
    _fake_embed.calls.append(len(texts))
    return vecs


_fake_embed.calls = []


def _write_kb(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


KB_ROWS = [
    {"uid": "a", "original_prompt": "Sort a list of numbers in ascending order.", "solution_code": "def s(x):\n    return sorted(x)",
     "test_list": ["xs = [3, 1]", "assert s(xs) == [1, 3]", "assert s([]) == []"], "source_file": "synthetic_llm", "source_task_id": "synllm_1_x"},
    {"uid": "b", "original_prompt": "sort a list of numbers  in ascending order", "solution_code": "def s2(x):\n    return sorted(x)",
     "test_list": ["assert s2([2, 1]) == [1, 2]"], "source_file": "synthetic_llm"},
    {"uid": "c", "original_prompt": "Reverse a string.", "solution_code": "def r(s):\n    return s[::-1]",
     "test_list": ["assert r('ab') == 'ba'"], "source_file": "synthetic_llm", "tests_modified": True},
    {"uid": "d", "original_prompt": "Write a function to find the shared elements from the given two lists.",
     "solution_code": "def similar_elements(a, b):\n    return tuple(set(a) & set(b))", "test_list": [],
     "source_file": "mbpp.jsonl", "source_task_id": 2},
    {"uid": "e", "original_prompt": "Sort a list of numbers in descending order.", "solution_code": "def sd(x):\n    return sorted(x, reverse=True)",
     "test_list": ["assert sd([1, 2]) == [2, 1]"], "source_file": "synthetic_llm"},
]


def test_retriever_index_filters_and_formats():
    import agent0.retrieval.retriever as rmod

    orig = rmod.get_embeddings_batch
    try:
        rmod.get_embeddings_batch = _fake_embed
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp) / "knowledge_base.jsonl"
            _write_kb(kb, KB_ROWS)
            r = rmod.KnowledgeRetriever(str(kb), exclude_source_task_ids=["Mbpp/2"])
            assert r.build_index() == 3 and r.stats["duplicates"] == 1 and r.stats["excluded_source"] == 1, r.stats
            assert r.stats["embedded_new"] == 3
            r2 = rmod.KnowledgeRetriever(str(kb), exclude_source_task_ids=["Mbpp/2"])
            r2.build_index()
            assert r2.stats["embedded_new"] == 0 and r2.stats["embedded_cached"] == 3, "index phải dùng lại theo hash"
            rows = [dict(x) for x in KB_ROWS]
            rows[2]["original_prompt"] = "Reverse the words of a sentence."
            _write_kb(kb, rows)  # cùng số dòng, khác nội dung
            r3 = rmod.KnowledgeRetriever(str(kb), exclude_source_task_ids=["Mbpp/2"])
            r3.build_index()
            assert r3.stats["embedded_new"] == 1 and r3.stats["embedded_cached"] == 2, r3.stats
            _write_kb(kb, KB_ROWS)

            r4 = rmod.KnowledgeRetriever(str(kb), exclude_test_repaired=True,
                                         exclude_prompts=["Write a function to find the shared elements from the given two lists."])
            r4.build_index()
            assert r4.stats["excluded_test_repaired"] == 1 and r4.stats["excluded_prompt"] == 1, r4.stats

            hits = r2.query("sort numbers ascending", n=2)
            assert hits[0]["uid"] == "a" and len(hits) == 2
            assert r2.query("sort numbers ascending", n=2, min_similarity=0.99) == [], "dưới ngưỡng -> zero-shot"
            plain = r2.query("sort a list of numbers", n=2)
            assert {h["uid"] for h in plain} == {"a", "e"}, [h["uid"] for h in plain]
            mmr = r2.query("sort a list of numbers", n=2, mmr_lambda=0.3)
            assert {h["uid"] for h in mmr} != {"a", "e"} and "c" in {h["uid"] for h in mmr}, "MMR phải tránh hai ví dụ na ná nhau"

            text = r2.format_few_shot(hits[:1], style="tests")
            assert '"""' in text and "xs = [3, 1]" in text and "assert s(xs) == [1, 3]" in text and "assert s([])" not in text, text
            assert "Reasoning" not in r2.format_few_shot(hits[:1], style="code")
    finally:
        rmod.get_embeddings_batch = orig


def test_generate_with_selector_workers_and_resume_guard():
    import agent0.retrieval.retriever as rmod

    orig_embed, orig_chat = rmod.get_embeddings_batch, rb.chat_completion
    prompts = []

    def fake_chat(runtime, messages, **kw):
        prompts.append(messages[-1]["content"])
        return fake_llm()(runtime, messages, **kw)

    try:
        rmod.get_embeddings_batch = _fake_embed
        rb.chat_completion = fake_chat
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp) / "knowledge_base.jsonl"
            _write_kb(kb, KB_ROWS)
            retriever = rb.build_retriever(kb, "m", "http://x", exclude_tasks=FAKE_TASKS)
            sel = rb.FewShotSelector(retriever, 2, "retrieved", "tests")
            rt = RuntimeConfig(provider="ollama", model="m", backend="b", base_url="http://x")
            out = Path(tmp) / "m" / "with_kb_n2_samples.jsonl"
            stats = rb.generate_all(rt, FAKE_TASKS, out, sel, workers=3)
            assert stats["generated"] == 3 and stats["errors"] == 0 and stats["retrieval"]["tasks"] == 3, stats
            assert sorted(r["task_id"] for r in rb.read_jsonl(out)) == FAKE_IDS
            log = rb.read_jsonl(rb.sidecar(out, "retrieval.jsonl"))
            assert len(log) == 3 and all(x["n"] == 2 for x in log)
            assert all("### Example 1" in p for p in prompts)

            rnd = rb.FewShotSelector(retriever, 2, "random", seed=1)
            assert rnd.select(FAKE_TASKS[0])[1]["uids"] == rnd.select(FAKE_TASKS[0])[1]["uids"], "random phải tái lập theo seed"
            fixed = rb.FewShotSelector(retriever, 2, "fixed", seed=1)
            assert fixed.select(FAKE_TASKS[0])[1]["uids"] == fixed.select(FAKE_TASKS[1])[1]["uids"]

            args = argparse_ns(mode="with_kb", n_examples=2, prompt_style="evalplus", temperature=0.0, seed=0,
                               max_tokens=1800, limit=None, embed_model="m")
            cfg = rb.generation_config(args, "with_kb", kb, sel)
            rb.write_meta(out, rt, cfg, stats, rb.now_iso())
            rb.check_resume_compatible(out, rt, cfg)  # cùng cấu hình -> được
            other = rb.generation_config(argparse_ns(**{**vars(args), "temperature": 0.7}), "with_kb", kb, sel)
            expect_raises(SystemExit, rb.check_resume_compatible, out, rt, other)
            no_meta = Path(tmp) / "old_samples.jsonl"
            no_meta.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
            expect_raises(SystemExit, rb.check_resume_compatible, no_meta, rt, cfg)
            rb.check_resume_compatible(no_meta, rt, cfg, force=True)
            assert rb.fewshot_variant(argparse_ns(example_selection="random", fewshot_format="full", min_similarity=0.6)) == "_random_full_tau0.6"
    finally:
        rmod.get_embeddings_batch, rb.chat_completion = orig_embed, orig_chat


def argparse_ns(**kw):
    import argparse

    return argparse.Namespace(**kw)


def test_llm_response_cache():
    rt = RuntimeConfig(provider="ollama", model="m", backend="b", base_url="http://localhost:11434")
    calls = []
    orig_post = llm_client.http_post_json

    def fake_post(url, payload, timeout=300, headers=None):
        calls.append(payload)
        return {"message": {"content": f"answer {len(calls)}"}}

    try:
        llm_client.http_post_json = fake_post
        with tempfile.TemporaryDirectory() as tmp:
            llm_client.configure_cache(tmp)
            llm_client.reset_call_stats()
            msgs = [{"role": "user", "content": "hi"}]
            a = llm_client.chat_completion(rt, msgs, temperature=0.0, seed=0)
            b = llm_client.chat_completion(rt, msgs, temperature=0.0, seed=0)
            assert a == b == "answer 1" and len(calls) == 1
            c = llm_client.chat_completion(rt, msgs, temperature=0.8)          # lấy mẫu không seed -> không cache
            d = llm_client.chat_completion(rt, msgs, temperature=0.8)
            assert c != d and len(calls) == 3
            e = llm_client.chat_completion(rt, msgs, temperature=0.8, seed=5)  # có seed -> cache
            f = llm_client.chat_completion(rt, msgs, temperature=0.8, seed=5)
            assert e == f and len(calls) == 4
            st = llm_client.call_stats()
            assert st["calls"] == 6 and st["api_calls"] == 4 and st["cache_hits"] == 2, st
    finally:
        llm_client.http_post_json = orig_post
        llm_client.configure_cache(None)
        llm_client.reset_call_stats()


def test_run_curriculum_offline_mbpp():
    """offline + nguồn mbpp: KB từ lời giải mẫu; bỏ bài MBPP+; khử ô nhiễm; uid duy nhất; chạy lại -> bỏ qua đã có."""
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "PYTHONPATH": str(SOURCE_CODE)}
        cmd = [sys.executable, "-m", "agent0.scripts.run_curriculum", "--provider", "offline", "--source", "mbpp",
               "--limit", "60", "--items_per_strategy", "4", "--output_dir", tmp, "--log_level", "WARNING"]
        for _ in range(2):
            proc = subprocess.run(cmd, cwd=SOURCE_CODE, capture_output=True, text=True, env=env, timeout=600)
            assert proc.returncode == 0, proc.stderr[-2000:]
        d = Path(tmp) / "mbpp_seeded" / "reference"
        summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        st = summary["stats"]
        rows = [json.loads(x) for x in (d / "knowledge_base.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        plus = {int(t.split("/")[1]) for t in rb.load_expected_task_ids()}
        assert rows and not any(int(r["source_task_id"]) in plus for r in rows), "KB không được chứa bài MBPP+"
        assert len({r["uid"] for r in rows}) == len(rows) and all(r["solution_source"] == "reference" for r in rows)
        assert st["excluded_mbpp_plus"] > 0 and st["processed"] == 0 and st["skipped_existing_kb"] > 0, st
        assert sum(1 for _ in (d / "summary_history.jsonl").open(encoding="utf-8")) == 2


def _synthetic_fake_llm():
    """LLM giả cho run_curriculum nguồn tự sinh: planner 2 subtopic, 4 đề; add_two có p_hat = 0.5, square/rev quá dễ,
    square có một test sai (sửa bằng đồng thuận), rectangle_area trùng một bài MBPP+."""
    code = {
        "add_two": ("def add_two(a, b):\n    return a + b\n", "def add_two(a, b):\n    return a - b\n"),
        "square": ("def square(x):\n    return x * x\n", "def square(x):\n    return x + x\n"),
        "rev": ("def rev(s):\n    return s[::-1]\n", "def rev(s):\n    return s\n"),
        "rectangle_area": ("def rectangle_area(l, w):\n    return l * w\n", None),
    }
    tasks = {
        "Arithmetic": [
            {"prompt": "Write a function `add_two(a, b)` that returns the sum of two integers a and b.", "function_name": "add_two",
             "test_list": ["assert add_two(1, 2) == 3", "assert add_two(-1, 1) == 0", "assert add_two(10, 5) == 15"]},
            {"prompt": "Write a function `square(x)` that returns x multiplied by itself.", "function_name": "square",
             "test_list": ["assert square(2) == 4", "assert square(3) == 9", "assert square(4) == 15"]},
        ],
        "Strings": [
            {"prompt": "Write a function `rev(s)` that returns the string s reversed character by character.", "function_name": "rev",
             "test_list": ["assert rev('abc') == 'cba'", "assert rev('') == ''", "assert rev('ab') == 'ba'"]},
            {"prompt": "Write a function to find the area of a rectangle.", "function_name": "rectangle_area",
             "test_list": ["assert rectangle_area(2, 3) == 6", "assert rectangle_area(1, 1) == 1", "assert rectangle_area(0, 5) == 0"]},
        ],
    }
    planner_prompts = []

    def fake(runtime, messages, temperature=0.2, max_tokens=2048, seed=None, **kw):
        sysmsg = messages[0]["content"] if messages[0]["role"] == "system" else ""
        user = messages[-1]["content"]
        if "curriculum designer" in sysmsg:
            planner_prompts.append(user)
            return "[]" if "MISSING" in user else json.dumps([
                {"name": "Arithmetic", "description": "arithmetic", "difficulty": "easy", "task_count": 2},
                {"name": "Strings", "description": "strings", "difficulty": "medium", "task_count": 2}])
        if "programming instructor" in sysmsg:
            return json.dumps(tasks[json.loads(user)["topic"]])
        if "coding tutor" in sysmsg:
            return "- direct computation"
        if "judge" in sysmsg:
            return "TEST"
        fn = next(f for f in code if f"{f}(" in user or f"`{f}" in user)
        good, bad = code[fn]
        if "from scratch" in user and seed is not None and seed >= 3000:  # mẫu p_hat (--seed 1)
            fake.p_hat_prompts.append(user)
            return bad if (fn == "add_two" and seed % 2) else good
        return good

    fake.p_hat_prompts = []
    return fake, planner_prompts


def _curriculum_modules():
    import agent0.curriculum.generator as gen
    import agent0.curriculum.planner as planner
    import agent0.curriculum.strategies as strategies

    return (gen, planner, agent_mod, strategies)


def test_run_curriculum_synthetic_fake_llm():
    """Nguồn tự sinh với LLM giả: đồng thuận sửa test, p_hat lọc đề quá dễ, khử ô nhiễm, phản hồi cho planner."""
    import agent0.scripts.run_curriculum as rc

    fake, planner_prompts = _synthetic_fake_llm()
    mods = _curriculum_modules()
    saved = [m.chat_completion for m in mods]
    argv = sys.argv
    try:
        for m in mods:
            m.chat_completion = fake
        with tempfile.TemporaryDirectory() as tmp:
            base = ["run_curriculum", "--provider", "ollama", "--model", "fake:1b", "--synthetic_count", "4", "--seed", "1",
                    "--output_dir", tmp, "--log_level", "ERROR", "--uncertainty_samples", "4"]
            sys.argv = base
            rc.main()
            d = Path(tmp) / "fake_1b"
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            assert s["stats"]["accepted"] == 1 and s["stats"]["skipped_contaminated"] == 1, s["stats"]
            assert s["stats"]["consensus_outcomes"] == {"tests_fixed": 1} and s["stats"]["reject_stages"]["p_hat"] == 2, s["stats"]
            assert s["uncertainty"]["in_band"] == 1 and s["uncertainty"]["too_easy"] == 2, s["uncertainty"]
            kb_rows = [json.loads(x) for x in (d / "knowledge_base.jsonl").read_text(encoding="utf-8").splitlines()]
            assert kb_rows[0]["function_name"] == "add_two" and kb_rows[0]["p_hat"] == 0.5
            rej = [json.loads(x) for x in (d / "rejected.jsonl").read_text(encoding="utf-8").splitlines()]
            sq = next(r for r in rej if "square" in r["original_prompt"])
            assert sq["reject_stage"] == "p_hat" and "assert square(4) == 16" in sq["test_list"], sq
            sys.argv = base + ["--feedback_from", str(d / "summary.json"), "--skip_previously_rejected"]
            planner_prompts.clear()
            rc.main()
            assert any("Feedback from the previous round" in p and "Strings" in p for p in planner_prompts)
            s2 = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            assert s2["stats"]["processed"] == 0 and s2["stats"]["skipped_existing_kb"] == 1, s2["stats"]
        with tempfile.TemporaryDirectory() as tmp:  # song song cho cùng kết quả
            sys.argv = ["run_curriculum", "--provider", "ollama", "--model", "fake:1b", "--synthetic_count", "4", "--seed", "1",
                        "--output_dir", tmp, "--log_level", "ERROR", "--uncertainty_samples", "4", "--workers", "3"]
            rc.main()
            s3 = json.loads((Path(tmp) / "fake_1b" / "summary.json").read_text(encoding="utf-8"))
            assert s3["stats"]["accepted"] == 1 and s3["stats"]["reject_stages"]["p_hat"] == 2, s3["stats"]
    finally:
        sys.argv = argv
        for m, f in zip(mods, saved):
            m.chat_completion = f


def test_p_hat_with_kb_and_split():
    """p_hat đo trên model + KB (đồng tiến hoá không huấn luyện) và tách KB theo p_hat cho thí nghiệm đối chứng."""
    import agent0.retrieval.retriever as rmod
    import agent0.scripts.run_curriculum as rc
    from agent0.curriculum import uncertainty as unc

    sys.path.insert(0, str(SOURCE_CODE / "tools"))
    import split_kb_by_phat as skb

    orig_embed, orig_chat = rmod.get_embeddings_batch, agent_mod.chat_completion
    argv = sys.argv
    mods = _curriculum_modules()
    saved = [m.chat_completion for m in mods]
    try:
        rmod.get_embeddings_batch = _fake_embed
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp) / "kb0" / "knowledge_base.jsonl"
            kb.parent.mkdir()
            _write_kb(kb, KB_ROWS)
            retriever = rmod.KnowledgeRetriever(str(kb), exclude_test_repaired=True)
            retriever.build_index()

            # 1) Cùng một đề: không có KB thì model giả luôn sai, có ví dụ KB thì luôn đúng -> p_hat 0 -> 1
            rec = {"task_id": "u", "prompt": "Add two numbers", "challenge_test_list": [],
                   "test_list": ["assert add(1, 2) == 3", "assert add(2, 2) == 4"]}
            agent_mod.chat_completion = lambda rt, messages, **kw: "```python\n" + (
                ADD_GOOD if "solved example tasks" in messages[-1]["content"] else ADD_BAD) + "\n```"
            text, info = unc.kb_few_shot(retriever, {"prompt": '"""\nSort numbers ascending.\nassert s([2]) == [2]\n"""'}, 2)
            assert info["n"] == 2 and info["uids"][0] == "a" and "### Example 1" in text and info["kb"] == str(kb), info
            assert unc.kb_few_shot(retriever, rec, 0) == ("", {"kb": str(kb), "n": 0, "uids": [], "similarities": []})
            bare = unc.estimate_pass_rate(RT_FAKE, rec, ADD_GOOD, k=2, timeout=10)
            with_kb = unc.estimate_pass_rate(RT_FAKE, rec, ADD_GOOD, k=2, timeout=10, few_shot=text)
            assert bare["p_hat"] == 0.0 and not bare["with_kb"] and with_kb["p_hat"] == 1.0 and with_kb["with_kb"]
            agent_mod.chat_completion = orig_chat

            # 2) run_curriculum --p_hat_kb --p_hat_action tag: mọi đề đã kiểm chứng được nhận, mỗi mục ghi p_hat + ví dụ KB
            fake, _ = _synthetic_fake_llm()
            for m in mods:
                m.chat_completion = fake
            out = Path(tmp) / "out"
            sys.argv = ["run_curriculum", "--provider", "ollama", "--model", "fake:1b", "--synthetic_count", "4", "--seed", "1",
                        "--output_dir", str(out), "--log_level", "ERROR", "--uncertainty_samples", "4",
                        "--p_hat_action", "tag", "--p_hat_kb", str(kb), "--p_hat_n_examples", "2", "--no_global_kb"]
            rc.main()
            d = out / "fake_1b"
            s = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            assert s["config"]["p_hat_kb"]["entries"] == 3 and s["config"]["p_hat_kb"]["filter"]["excluded_test_repaired"] == 1
            assert s["uncertainty"]["with_kb"] and s["stats"]["accepted"] == 3, (s["uncertainty"], s["stats"])
            rows = [json.loads(x) for x in (d / "knowledge_base.jsonl").read_text(encoding="utf-8").splitlines()]
            assert all(r["uncertainty"]["with_kb"] and r["uncertainty"]["kb_examples"]["n"] == 2 for r in rows)
            assert fake.p_hat_prompts and all("solved example tasks" in p for p in fake.p_hat_prompts)
            assert sorted(r["p_hat"] for r in rows) == [0.5, 1.0, 1.0], [r["p_hat"] for r in rows]

            # 3) tách KB theo p_hat: phat (trong khoảng) và ctrl (cùng cỡ, ngẫu nhiên)
            phat, ctrl, st = skb.split(rows)
            assert [r["function_name"] for r in phat] == ["add_two"] and len(ctrl) == 1, st
            assert st["tagged"] == 2 and st["dropped_test_repaired"] == 1, "mục test bị sửa (square) phải bị bỏ như benchmark"
            phat2, ctrl2, st2 = skb.split(rows, control="out_of_band")
            assert ctrl2[0]["p_hat"] == 1.0 and st2["overlap"] == 0
            sys.argv = ["split_kb_by_phat", str(d / "knowledge_base.jsonl"), "--out_dir", str(Path(tmp) / "split"), "--min_size", "1"]
            assert skb.main() == 0
            assert (Path(tmp) / "split" / "phat" / "knowledge_base.jsonl").exists()
            rep = json.loads((Path(tmp) / "split" / "split_report.json").read_text(encoding="utf-8"))
            assert rep["in_band"] == rep["control"] == 1
            sys.argv = ["split_kb_by_phat", str(kb), "--out_dir", str(Path(tmp) / "split0")]
            assert skb.main() == 2, "KB không có p_hat phải báo lỗi"
    finally:
        sys.argv = argv
        rmod.get_embeddings_batch, agent_mod.chat_completion = orig_embed, orig_chat
        for m, f in zip(mods, saved):
            m.chat_completion = f


# ───────────────────── Kaggle / nhiều model: extra_body, hết quota, run_matrix, summarize, đóng gói ─────────────────────

def test_extra_body_quota_and_groq_detection():
    assert llm_client.merge_extra_body({"a": 1, "options": {"seed": 0}, "max_tokens": 5},
                                       {"options": {"num_ctx": 8192}, "max_tokens": None, "b": 2}) == \
        {"a": 1, "options": {"seed": 0, "num_ctx": 8192}, "b": 2}
    sent, calls = [], []
    orig_post, orig_sleep = llm_client.http_post_json, llm_client.time.sleep

    def fake_post(url, payload, timeout=300, headers=None):
        sent.append((url, payload))
        if payload.get("model") == "quota":
            calls.append(1)
            raise llm_client.HTTPStatusError(429, url, "Rate limit reached on tokens per day (TPD): Limit 200000", 7.0)
        if url.endswith("/api/chat"):
            return {"message": {"content": "ok"}}
        return {"choices": [{"message": {"content": "ok"}}]}

    try:
        llm_client.http_post_json = fake_post
        llm_client.time.sleep = lambda s: (_ for _ in ()).throw(AssertionError("không được chờ khi hết quota ngày"))
        rt = RuntimeConfig(provider="groq", model="openai/gpt-oss-120b", backend="b", base_url="http://x/v1", api_key="k",
                           extra_body={"reasoning_effort": "low"})
        llm_client.chat_completion(rt, [{"role": "user", "content": "hi"}], temperature=0, seed=0)
        assert sent[-1][1]["reasoning_effort"] == "low" and sent[-1][1]["seed"] == 0
        rt2 = RuntimeConfig(provider="ollama", model="m", backend="b", base_url="http://x",
                            extra_body={"options": {"num_ctx": 8192}})
        llm_client.chat_completion(rt2, [{"role": "user", "content": "hi"}], temperature=0, seed=1)
        assert sent[-1][1]["options"] == {"temperature": 0, "num_predict": 2048, "seed": 1, "num_ctx": 8192}, sent[-1][1]
        rq = RuntimeConfig(provider="groq", model="quota", backend="b", base_url="http://x/v1", api_key="k")
        expect_raises(llm_client.QuotaExhaustedError, llm_client.chat_completion, rq, [{"role": "user", "content": "x"}])
        assert len(calls) == 1, "hết quota ngày phải báo lỗi ngay, không thử lại"
    finally:
        llm_client.http_post_json, llm_client.time.sleep = orig_post, orig_sleep
    with EnvPatch(GROQ_API_KEY="k"):
        assert resolve_runtime("auto", "openai/gpt-oss-120b").provider == "groq"
        assert resolve_runtime("auto", "qwen/qwen3.6-27b").provider == "groq"
    assert config.DEFAULT_MODEL == "openai/gpt-oss-120b"


def test_generate_stops_early_on_quota():
    calls = []

    def quota_chat(runtime, messages, **kw):
        calls.append(1)
        raise llm_client.QuotaExhaustedError("hết hạn mức")

    orig = rb.chat_completion
    try:
        rb.chat_completion = quota_chat
        with tempfile.TemporaryDirectory() as tmp:
            rt = RuntimeConfig(provider="ollama", model="m", backend="b", base_url="http://x")
            st = rb.generate_all(rt, FAKE_TASKS, Path(tmp) / "s.jsonl", max_consecutive_errors=10)
            assert len(calls) == 1 and st["stopped_early"] and st["missing"] == 3 and st["errors"] == 1, st
            rb.chat_completion = fake_llm(fail_ids=set(FAKE_IDS))
            st = rb.generate_all(rt, FAKE_TASKS, Path(tmp) / "t.jsonl", max_consecutive_errors=2)
            assert st["errors"] == 2 and st["missing"] == 3 and "liên tiếp" in st["stopped_early"], st
    finally:
        rb.chat_completion = orig


def test_matrix_roster_and_run_names():
    import argparse
    sys.path.insert(0, str(SOURCE_CODE / "tools"))
    import run_matrix as rm
    import summarize_matrix as sm

    roster = rm.load_roster(SOURCE_CODE / "kaggle" / "roster.json")
    labels = [m["label"] for m in roster]
    assert len(labels) == len(set(labels)) >= 8 and "gemma3-4b" in labels and "qwen3.5-397b" not in labels
    assert [m["label"] for m in rm.load_roster(SOURCE_CODE / "kaggle" / "roster.json", only=["qwen3.5-397b"])] == ["qwen3.5-397b"]
    by = {m["label"]: m for m in rm.load_roster(SOURCE_CODE / "kaggle" / "roster.json", include_disabled=True)}
    assert rm.is_local(by["gemma3-4b"]) and not rm.is_local(by["gpt-oss-120b"]) and not rm.is_local(by["qwen3.5-397b"])
    assert by["gpt-oss-120b"]["extra_body"] == {"reasoning_effort": "low"} and by["gemma3-4b"]["workers"] == 4
    assert rm.is_local({"provider": "custom", "base_url": "http://127.0.0.1:8000/v1"})
    with tempfile.TemporaryDirectory() as tmp:
        args = argparse.Namespace(results_dir=tmp, time_budget_hours=1, configs=list(rm.CONFIGS), n_examples=2,
                                  kb_path="kb.jsonl", eval="skip", fewshot_format="tests", min_similarity=0.0,
                                  kb_tag=None, exclude_test_repaired=False)
        m = rm.Matrix(args, [by["gemma3-4b"]])
        names = [m.run_paths(by["gemma3-4b"], c)[1].name for c in rm.CONFIGS]
        assert names == ["baseline_samples.jsonl", "with_kb_n2_samples.jsonl", "with_kb_n2_random_samples.jsonl",
                         "with_kb_n2_fixed_samples.jsonl"], names
        assert (Path(tmp) / "matrix_status.json").exists()
        assert [sm.parse_run(n[: -len("_samples.jsonl")])[:2] for n in names] == \
            [("baseline", None), ("retrieved", 2), ("random", 2), ("fixed", 2)]
    with tempfile.TemporaryDirectory() as tmp:  # bỏ mục test bị sửa + nhãn KB: tên lần chạy và lệnh sinh
        args = argparse.Namespace(results_dir=tmp, time_budget_hours=1, configs=list(rm.CONFIGS), n_examples=2,
                                  kb_path="kb.jsonl", eval="skip", fewshot_format="tests", min_similarity=0.0,
                                  kb_tag="phat", exclude_test_repaired=True, prompt_style="evalplus", temperature=0.0,
                                  seed=0, max_tokens=1800, workers=4, max_consecutive_errors=10, limit=None,
                                  embed_model="mxbai-embed-large", embed_base_url="http://localhost:11434")
        m = rm.Matrix(args, [by["gemma3-4b"]])
        names = [m.run_paths(by["gemma3-4b"], c)[1].name for c in rm.CONFIGS]
        assert names == ["baseline_samples.jsonl", "with_kb_n2_notr_kbphat_samples.jsonl",
                         "with_kb_n2_random_notr_kbphat_samples.jsonl", "with_kb_n2_fixed_notr_kbphat_samples.jsonl"], names
        assert sm.parse_run("with_kb_n2_random_notr_kbphat") == ("random", 2, "notr_kbphat")
        cmd = m.generate_cmd(by["gemma3-4b"], "retrieved", Path(tmp) / "x_samples.jsonl")
        assert "--exclude_test_repaired" in cmd and "--example_selection" in cmd
        assert "--exclude_test_repaired" not in m.generate_cmd(by["gemma3-4b"], "baseline", Path(tmp) / "b_samples.jsonl")
    local = [m for m in rm.load_roster(SOURCE_CODE / "kaggle" / "roster.json") if rm.is_local(m)]
    lanes = rm.split_lanes(local, 2)
    loads = [sum(m["params_b"] for m in g) for g in lanes]
    assert len(lanes) == 2 and sorted(m["label"] for g in lanes for m in g) == sorted(m["label"] for m in local)
    assert abs(loads[0] - loads[1]) <= 2, loads
    assert rm.split_lanes(local[:1], 2) == [local[:1]] and rm.split_lanes([], 2) == []
    assert sm.parse_run("with_kb_n3_full") == ("retrieved", 3, "full")
    assert sm.describe_variant("notr_kbphat") == "bỏ mục test bị sửa, KB phat"
    assert sm.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert sm.spearman([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], n_perm=200)[0] == 1.0


def _fake_matrix_run(folder: Path, name: str, expected, passes, sims=None):
    """Ghi một lần chạy giả đủ 378 bài đã chấm: samples, metrics, eval_results (+ log truy hồi nếu có sims)."""
    folder.mkdir(parents=True, exist_ok=True)
    samples = folder / f"{name}_samples.jsonl"
    samples.write_text("".join(json.dumps({"task_id": t, "solution": "def f(x):\n    return x\n"}) + "\n" for t in expected),
                       encoding="utf-8")
    ev = {t: [{"base_status": "pass" if passes(i) else "fail", "plus_status": "pass" if passes(i) else "fail"}]
          for i, t in enumerate(expected)}
    rb.sidecar(samples, "eval_results.json").write_text(json.dumps({"eval": ev}), encoding="utf-8")
    rate = round(100 * sum(1 for i in range(len(expected)) if passes(i)) / len(expected), 2)
    rb.sidecar(samples, "metrics.json").write_text(json.dumps({"base_pass@1": rate, "plus_pass@1": rate}), encoding="utf-8")
    if sims is not None:
        rb.sidecar(samples, "retrieval.jsonl").write_text(
            "".join(json.dumps({"task_id": t, "n": 2, "uids": ["a", "b"], "similarities": [sims(i), sims(i) / 2]}) + "\n"
                    for i, t in enumerate(expected)), encoding="utf-8")


def test_summarize_matrix_variants_and_similarity():
    sys.path.insert(0, str(SOURCE_CODE / "tools"))
    import summarize_matrix as sm

    expected = rb.load_expected_task_ids()
    n = len(expected)
    sim = lambda i: round(i / n, 4)  # noqa: E731 - bài càng về sau ví dụ truy hồi càng giống
    base = lambda i: i % 2 == 0  # noqa: E731
    argv = sys.argv
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for label in ("m1", "m2"):
                d = Path(tmp) / label
                _fake_matrix_run(d, "baseline", expected, base)
                # retrieved chỉ hơn baseline ở nhóm giống nhất; random không đổi gì
                _fake_matrix_run(d, "with_kb_n2_notr", expected, lambda i: base(i) or sim(i) >= 0.8, sims=sim)
                _fake_matrix_run(d, "with_kb_n2_random_notr", expected, base)
                _fake_matrix_run(d, "with_kb_n2_notr_kbctrl", expected, base, sims=sim)
                _fake_matrix_run(d, "with_kb_n2_notr_kbphat", expected, lambda i: base(i) or i % 7 == 0, sims=sim)
            d = Path(tmp) / "m1-grpo"  # model sau RL: hơn m1 ở baseline, bằng ở retrieved
            _fake_matrix_run(d, "baseline", expected, lambda i: base(i) or i % 5 == 0)
            _fake_matrix_run(d, "with_kb_n2_notr", expected, lambda i: base(i) or sim(i) >= 0.8, sims=sim)
            sys.argv = ["summarize_matrix", "--results_dir", tmp, "--model_pairs", "m1:m1-grpo", "m1:khong-co"]
            assert sm.main() == 0
            out = Path(tmp) / "summary"
            md = (out / "matrix_summary.md").read_text(encoding="utf-8")
            fams = {r["family"] for r in __import__("csv").DictReader((out / "matrix_comparisons.csv").open(encoding="utf-8"))}
            assert {"baseline -> retrieved (n=2, notr)", "random -> retrieved (n=2, notr)",
                    "retrieved: notr -> notr_kbctrl (n=2)", "retrieved: notr_kbctrl -> notr_kbphat (n=2)",
                    "model: m1 -> m1-grpo (baseline)", "model: m1 -> m1-grpo (retrieved, n=2, notr)"} <= fams, fams
            pair = {(r["family"], r["metric"]): r for r in __import__("csv").DictReader(
                (out / "matrix_comparisons.csv").open(encoding="utf-8"))}
            assert float(pair[("model: m1 -> m1-grpo (baseline)", "plus")]["delta_pp"]) > 0
            assert float(pair[("model: m1 -> m1-grpo (retrieved, n=2, notr)", "plus")]["delta_pp"]) == 0
            assert "retrieved [bỏ mục test bị sửa]" in md and "retrieved [bỏ mục test bị sửa, KB phat]" in md
            assert "Δ theo độ giống" in md and "gần trùng (≥ 0.9)" in md
            rows = list(__import__("csv").DictReader((out / "matrix_similarity.csv").open(encoding="utf-8")))
            main_rows = [r for r in rows if r["variant"] == "notr" and r["label"] == "m1"]
            by_stratum = {r["stratum"].split(" ")[0]: float(r["delta_retrieved"]) for r in main_rows}
            assert by_stratum["thấp"] == 0 and by_stratum["cao"] > 0 and by_stratum["gần"] > 0, by_stratum
            assert all(float(r["delta_random"]) == 0 for r in main_rows), main_rows
    finally:
        sys.argv = argv


def test_grpo_data_and_reward():
    """Dữ liệu RL từ KB (prompt dạng MBPP+, test chuẩn hoá, khử ô nhiễm, uid duy nhất) và thưởng bằng harness."""
    from agent0.training.data import frontier_filter, load_rl_tasks, split_train_val
    from agent0.training.rewards import HarnessReward, adpo_scale, completion_text

    good = "def add_two(a, b):\n    return a + b"
    kb_rows = [
        {"uid": "u1", "original_prompt": "Add two integers and return the sum.", "solution_code": good,
         "test_list": ["assert add_two(1, 2) == 3", "assert add_two(0, 0) == 0", "assert add_two(-1, 1) == 0"],
         "challenge_test_list": ["assert add_two(10, 5) == 15"], "test_setup_code": "def add_two(a, b):\n    pass"},
        {"uid": "u1", "original_prompt": "Sort a list ascending with selection sort.",  # uid trùng, đề khác
         "test_list": ["sel_sort([3, 1, 2]) == [1, 2, 3]", "sel_sort([]) == []", "sel_sort([1]) == [1]"]},
        {"uid": "u3", "original_prompt": "Make a Rectangle class with area().",
         "test_list": ["r = Rectangle(2, 3)\nassert r.area() == 6", "r = Rectangle(1, 1)\nassert r.area() == 1",
                       "r = Rectangle(0, 5)\nassert r.area() == 0"]},
        {"uid": "u4", "original_prompt": "Reverse a string.", "tests_modified": True,
         "test_list": ["assert rev('ab') == 'ba'", "assert rev('') == ''", "assert rev('a') == 'a'"]},
        {"uid": "u5", "original_prompt": "Add two integers and return the sum!",  # trùng đề u1
         "test_list": ["assert add_two(1, 2) == 3", "assert add_two(2, 2) == 4", "assert add_two(3, 3) == 6"]},
        {"uid": "u6", "original_prompt": "Write a function to find the similar elements from the given two tuple lists.",
         "test_list": ["assert set(similar_elements((3, 4), (4, 5))) == {4}", "assert similar_elements((), ()) == ()",
                       "assert set(similar_elements((1,), (1,))) == {1}"]},  # giống MBPP+ (Mbpp/2)
    ]
    with tempfile.TemporaryDirectory() as tmp:
        kb = Path(tmp) / "knowledge_base.jsonl"
        _write_kb(kb, kb_rows)
        rows, stats = load_rl_tasks([kb])
        assert stats["test_repaired"] == 1 and stats["duplicate"] == 1 and stats["contaminated"] == 1, stats
        by = {r["description"].split()[0]: r for r in rows}
        assert set(by) == {"Add", "Sort", "Make"} and len({r["uid"] for r in rows}) == 3
        assert by["Sort"]["tests"][0] == "assert sel_sort([3, 1, 2]) == [1, 2, 3]"  # so sánh trần -> assert
        user = by["Make"]["prompt"][-1]["content"]
        assert '"""\nMake a Rectangle class with area().\nr = Rectangle(2, 3)\nassert r.area() == 6\n"""' in user
        assert user.startswith("Please provide a self-contained Python script") and by["Make"]["prompt"][0]["role"] == "system"
        assert len(by["Add"]["tests"]) == 4 and by["Add"]["setup"].startswith("def add_two")
        assert split_train_val(rows, 0.5, 1) == split_train_val(list(rows), 0.5, 1)

        uids = [r["uid"] for r in rows]
        chosen, rule = frontier_filter(rows, {uids[0]: 0.5, uids[1]: 1.0, uids[2]: 0.0}, min_rows=1)
        assert len(chosen) == 1 and rule.startswith("0.25")
        chosen, rule = frontier_filter(rows, {uids[0]: 0.0, uids[1]: 1.0, uids[2]: 0.0}, min_rows=1)
        assert len(chosen) == 2 and rule == "p_hat < 1"

        add = by["Add"]
        reward = HarnessReward()
        comps = [[{"role": "assistant", "content": f"```python\n{good}\n```"}],
                 [{"role": "assistant", "content": "```python\ndef add_two(a, b):\n    return a - b\n```"}],
                 "Tôi không biết.", good]
        r = reward(completions=comps, tests=[add["tests"]] * 4, setup=[add["setup"]] * 4)
        assert r[0] == 1.1 and r[2] == 0.0 and r[3] == 1.0 and 0.1 <= r[1] < 1.1, r
        assert reward.history[-1]["all_pass"] == 0.5
        truncated = "```python\ndef add_two(a, b):\n    return (a +"  # hết max_completion giữa chừng
        r = reward(completions=[truncated], tests=[add["tests"]], setup=[add["setup"]])
        assert r == [0.0] and reward.history[-1]["has_code"] == 1.0 and reward.history[-1]["code_ok"] == 0.0, r
        rb_ = HarnessReward(mode="binary", adpo=True, s_min=0.5)
        r = rb_(completions=comps[:2], tests=[add["tests"]] * 2, setup=[add["setup"]] * 2, p_hat=[0.5, float("nan")])
        assert r == [1.1 * 0.75, 0.1], r
        assert adpo_scale(None) == 1.0 and adpo_scale(0.0) == 0.5 and adpo_scale(1.0) == 1.0
        assert completion_text({"content": "x"}) == "x" and completion_text(None) == ""


def _make_tiny_model(path: Path) -> bool:
    """Model Qwen2 ngẫu nhiên rất nhỏ + tokenizer ChatML, để chạy thử pipeline transformers/TRL trên CPU."""
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import GenerationConfig, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        import peft  # noqa: F401
        import trl  # noqa: F401
    except Exception:
        return False
    corpus = ["def f(x):\n    return x + 1\n", "```python\nprint('ok')\n```", "Please provide a self-contained Python script",
              "assert f(1) == 2", "Write a function to add numbers."] * 20
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<|endoftext|>", "<|im_start|>",
                            "<|im_end|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = ("{% for m in messages %}{{'<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n'}}"
                          "{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}")
    fast.save_pretrained(path)
    im_end = fast.convert_tokens_to_ids("<|im_end|>")
    cfg = Qwen2Config(vocab_size=len(fast), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=4096,
                      tie_word_embeddings=True, eos_token_id=im_end, pad_token_id=fast.pad_token_id)
    model = Qwen2ForCausalLM(cfg)
    model.generation_config = GenerationConfig(eos_token_id=[im_end], pad_token_id=fast.pad_token_id, do_sample=True,
                                               temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.05)
    model.save_pretrained(path)
    return True


def test_grpo_pipeline_tiny_model():
    """prepare (p_hat) -> train (GRPO + LoRA, 2 bước) -> merge -> score -> hf_generate, với model nhỏ trên CPU.
    Bỏ qua nếu máy không có torch / transformers / trl / peft."""
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "tiny"
        if not _make_tiny_model(model):
            print("   (bỏ qua: thiếu torch / transformers / trl / peft)")
            return
        kb = Path(tmp) / "kb.jsonl"
        _write_kb(kb, [{"uid": f"t{i}", "original_prompt": f"Return x plus {i} for an integer x.",
                        "test_list": [f"assert plus_{i}(1) == {1 + i}", f"assert plus_{i}(0) == {i}",
                                      f"assert plus_{i}(-1) == {i - 1}"]} for i in range(12)])
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
        tool = [sys.executable, "tools/grpo_executor.py"]

        def run(cmd):
            proc = subprocess.run(cmd, cwd=SOURCE_CODE, capture_output=True, text=True, env=env, timeout=900)
            assert proc.returncode == 0, f"{cmd[1:3]} lỗi:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}"
            return proc.stdout

        data = Path(tmp) / "data"
        out = run(tool + ["prepare", "--kb", str(kb), "--out_dir", str(data), "--model", str(model), "--p_hat_k", "2",
                          "--max_new_tokens", "8", "--min_train", "4", "--val_frac", "0.25", "--decontam", "0"])
        rep = json.loads((data / "prepare_report.json").read_text(encoding="utf-8"))
        assert rep["n_train"] >= 4 and rep["p_hat"]["k"] == 2 and "p_hat" in json.loads(
            (data / "train.jsonl").read_text(encoding="utf-8").splitlines()[0]), out
        run_dir = Path(tmp) / "run"
        run(tool + ["train", "--model", str(model), "--train", str(data / "train.jsonl"), "--out_dir", str(run_dir),
                    "--max_steps", "2", "--num_generations", "2", "--per_device_batch", "2", "--grad_accum", "2",
                    "--max_completion", "8", "--save_steps", "1", "--adpo", "--beta", "0.01"])
        summary = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
        assert summary["steps"] == 2 and (run_dir / "adapter" / "adapter_config.json").exists()
        assert any("harness" in json.loads(l) for l in (run_dir / "train_log.jsonl").read_text().splitlines())
        run(tool + ["merge", "--model", str(model), "--adapter", str(run_dir / "adapter"), "--out", str(Path(tmp) / "merged"),
                    "--dtype", "float32"])
        run(tool + ["score", "--model", str(model), "--adapter", str(run_dir / "adapter"), "--tasks",
                    str(data / "val.jsonl"), "--out", str(Path(tmp) / "val.json"), "--max_new_tokens", "8"])
        val = json.loads((Path(tmp) / "val.json").read_text(encoding="utf-8"))
        assert val["n"] == len((data / "val.jsonl").read_text().splitlines()) and len(val["per_task"]) == val["n"]

        from agent0.training.hf_backend import HFGenerator

        gen = HFGenerator(str(Path(tmp) / "merged"), max_new_tokens=6)
        gc = gen.model.generation_config  # không mang temperature 0.7 / top_k 20 / repetition_penalty của model
        assert gc.temperature in (None, 1.0) and gc.top_k in (None, 50) and gc.repetition_penalty in (None, 1.0), gc
        convs = [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "write code please"}]]
        greedy = gen.generate(convs, n=2, temperature=0.0)
        assert all(len(a) == 2 and a[0] == a[1] for a in greedy)
        assert all(len(a) == 3 for a in gen.generate(convs, n=3, temperature=1.0, top_p=0.9))

        try:  # hf_generate cần danh sách bài MBPP+ của evalplus
            from evalplus.data import get_mbpp_plus  # noqa: F401
        except Exception:
            return
        samples = Path(tmp) / "res" / "tiny-grpo" / "baseline_samples.jsonl"
        run([sys.executable, "tools/hf_generate.py", "--model", str(model), "--adapter", str(run_dir / "adapter"),
             "--output", str(samples), "--limit", "2", "--max_new_tokens", "8", "--batch_size", "1", "--chunk_tasks", "1"])
        rows = rb.read_jsonl(samples)
        assert [r["task_id"] for r in rows] == ["Mbpp/2", "Mbpp/3"] and all(r["solution"].strip() for r in rows)
        run([sys.executable, "tools/hf_generate.py", "--model", str(model), "--adapter", str(run_dir / "adapter"),
             "--output", str(samples), "--limit", "3", "--max_new_tokens", "8", "--resume"])
        assert [r["task_id"] for r in rb.read_jsonl(samples)] == ["Mbpp/2", "Mbpp/3", "Mbpp/4"]
        meta = json.loads(rb.sidecar(samples, "meta.json").read_text(encoding="utf-8"))
        assert meta["latest"]["config"]["adapter"] and meta["latest"]["config"]["temperature"] == 0.0


def test_package_for_kaggle():
    sys.path.insert(0, str(SOURCE_CODE / "tools"))
    import package_for_kaggle as pk

    ok = lambda p: pk.wanted(Path(p), include_results=False)  # noqa: E731
    assert ok("Source_Code/agent0/core/config.py") and ok("Data/mbpp/mbpp.jsonl") and ok(".env.example")
    assert ok("Source_Code/kaggle/cs2311_kaggle.ipynb") and ok("Data/knowledge_base/all_models_clean/knowledge_base.jsonl")
    for bad in (".env", "Source_Code/.env", "Source_Code/agent0/__pycache__/x.pyc", "Report/a.pdf", "Slide/b.pdf",
                "Data/results/benchmark/m/x_samples.jsonl", "Data/knowledge_base/all_models/knowledge_base.index.json"):
        assert not ok(bad), bad
    assert pk.wanted(Path("Data/results/benchmark/m/x_samples.jsonl"), include_results=True)


def main() -> int:
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} test qua")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
