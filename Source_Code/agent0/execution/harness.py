"""Harness chạy test trong tiến trình con cô lập — dùng chung cho pipeline sinh KB (sandbox.py),
tập dev của benchmark (strict_check.py) và bước sửa test theo đồng thuận (agent.py).

So với cách cũ (ghép lời giải + assert thành một script, PASS khi returncode == 0):
- Lời giải được nạp như module (__name__ = "solution"): khối `if __name__ == "__main__":` không chạy.
- exit()/sys.exit() khi nạp hoặc trong test tính là FAIL (không còn PASS giả).
- Từng dòng test chạy riêng theo thứ tự trong cùng namespace; dòng so sánh trần (`f(x) == y`)
  được hiểu là assert; assert sai dạng `assert L == R` ghi lại giá trị thực tế của L.
- Phải có ít nhất 1 assert và test phải gọi tới ít nhất một hàm/lớp do lời giải định nghĩa.
- Tiến trình con: `python -I`, biến môi trường sạch (không lộ API key), giới hạn CPU/RAM khi hệ điều
  hành cho phép, chặn kết nối mạng ở mức Python (socket.connect, getaddrinfo...).
- Thứ tự: test_setup_code -> lời giải -> test; nếu setup dùng tên của lời giải (kiểu MBPP) thì chạy lời giải trước.

Với mọi assert dạng `assert L == R`, L được tính đúng một lần và giá trị (repr) được ghi vào
'values' — dùng để so sánh nhiều lời giải với nhau ở bước sửa test theo đồng thuận.
"""
import json
import os
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional

_MARKER = "__AGENT0_HARNESS_RESULT__"

_HARNESS = r'''
import ast, contextlib, io, json, sys

def _limits(cpu, mem):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        try:
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass
    except Exception:
        pass

def _block_network():
    import socket
    def _no_net(*args, **kwargs):
        raise OSError("Network access is disabled in the Agent0 sandbox")
    socket.socket.connect = _no_net
    socket.socket.connect_ex = _no_net
    socket.create_connection = _no_net
    socket.getaddrinfo = _no_net
    socket.gethostbyname = _no_net

_limits(int(sys.argv[2]), int(sys.argv[3]))
_block_network()
with open(sys.argv[1], encoding="utf-8") as fh:
    payload = json.load(fh)
solution, setup, tests, mode = payload["solution"], payload["setup"], payload["tests"], payload["mode"]
sink = io.StringIO()

def _short(text, n=300):
    text = str(text)
    return text if len(text) <= n else text[:n] + "..."

def _run(code, g, name):
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        exec(compile(code, name, "exec"), g)

def _load():
    # Thứ tự cũ của pipeline: setup -> lời giải (setup của đề tự sinh thường là import hoặc khung hàm,
    # lời giải định nghĩa sau sẽ ghi đè). Nếu setup dùng tên do lời giải định nghĩa (kiểu MBPP:
    # `root = Node(1)`), chạy lại theo thứ tự MBPP: lời giải -> setup.
    def fresh():
        return {"__name__": "solution", "__file__": "solution.py"}
    orders = [("setup", "solution"), ("solution", "setup")] if setup.strip() else [("solution",)]
    codes = {"setup": setup, "solution": solution}
    last_err = None
    for attempt, order in enumerate(orders):
        g = fresh()
        try:
            for name in order:
                _run(codes[name], g, "<%s>" % name)
            return g, None
        except SystemExit:
            return None, "SystemExit while loading the solution (the code calls exit()/sys.exit())"
        except NameError as exc:
            last_err = "NameError: %s" % exc
            continue
        except BaseException as exc:
            return None, "%s: %s" % (type(exc).__name__, _short(exc))
    return None, last_err

def _classify(test):
    try:
        mod = ast.parse(test)
    except SyntaxError as exc:
        return "syntax_error", test, None
    if len(mod.body) == 1:
        node = mod.body[0]
        if isinstance(node, ast.Assert):
            return "assert", test, node
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Compare):
            fixed = "assert " + test.strip()
            return "converted", fixed, ast.parse(fixed).body[0]
    if any(isinstance(n, ast.Assert) for n in ast.walk(mod)):
        return "assert_block", test, None
    return "statement", test, None

def _eq_sides(node):
    t = node.test if isinstance(node, ast.Assert) else None
    if isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq):
        return ast.unparse(t.left), ast.unparse(t.comparators[0])
    return None, None

def _defined_names(code):
    try:
        mod = ast.parse(code)
    except SyntaxError:
        return []
    names = []
    for n in mod.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(n.name)
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.append(t.id)
    return names

def _referenced(tests, names):
    wanted = set(names)
    for test in tests:
        try:
            for n in ast.walk(ast.parse(test)):
                if isinstance(n, ast.Name) and n.id in wanted:
                    return True
                if isinstance(n, ast.Attribute) and n.attr in wanted:
                    return True
        except SyntaxError:
            continue
    return False

res = {"mode": mode, "loaded": False, "load_error": None, "n_asserts": 0, "n_ok": 0, "n_statements": 0,
       "converted": 0, "syntax_errors": 0, "failures": [], "statement_errors": [], "values": {}}
defined = _defined_names(solution)
res["defined_names"] = defined[:20]
res["referenced"] = _referenced(tests + [setup], defined) if defined else True
g, err = _load()
res["load_error"] = err
res["loaded"] = g is not None

def _eval(expr):
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        return eval(compile(expr, "<expr>", "eval"), g)

if g is not None:
    for idx, raw in enumerate(tests):
        kind, code, node = _classify(raw)
        if kind == "syntax_error":
            res["syntax_errors"] += 1
            res["failures"].append({"index": idx, "test": _short(raw, 200), "error": "SyntaxError in test"})
            continue
        if kind == "converted":
            res["converted"] += 1
        is_check = kind in ("assert", "converted", "assert_block")
        left = right = None
        if node is not None:
            left, right = _eq_sides(node)
        if is_check:
            res["n_asserts"] += 1
        else:
            res["n_statements"] += 1
        if left is not None:
            # `assert L == R`: tính L đúng một lần (như Python), ghi lại giá trị để so sánh nhiều lời giải
            try:
                v_left = _eval(left)
            except SystemExit:
                res["values"][str(idx)] = {"ok": False, "error": "SystemExit"}
                res["failures"].append({"index": idx, "test": _short(code, 200), "error": "SystemExit inside the test", "left": left})
                continue
            except BaseException as exc:
                msg = "%s: %s" % (type(exc).__name__, _short(exc, 200))
                res["values"][str(idx)] = {"ok": False, "error": msg}
                res["failures"].append({"index": idx, "test": _short(code, 200), "error": msg, "left": left})
                continue
            res["values"][str(idx)] = {"ok": True, "repr": _short(repr(v_left), 2000)}
            try:
                v_right = _eval(right)
                ok = bool(v_left == v_right)
            except SystemExit:
                res["failures"].append({"index": idx, "test": _short(code, 200), "error": "SystemExit inside the test", "left": left})
                continue
            except BaseException as exc:
                res["failures"].append({"index": idx, "test": _short(code, 200), "left": left,
                                        "error": "%s: %s" % (type(exc).__name__, _short(exc, 200))})
                continue
            if ok:
                res["n_ok"] += 1
            else:
                res["failures"].append({"index": idx, "test": _short(code, 200), "error": "AssertionError",
                                        "left": left, "right": right,
                                        "actual": _short(repr(v_left), 500), "expected": _short(repr(v_right), 500)})
            continue
        try:
            _run(code, g, "<test>")
            if is_check:
                res["n_ok"] += 1
        except SystemExit:
            entry = {"index": idx, "test": _short(code, 200), "error": "SystemExit inside the test"}
            (res["failures"] if is_check else res["statement_errors"]).append(entry)
        except BaseException as exc:
            entry = {"index": idx, "test": _short(code, 200), "error": "%s: %s" % (type(exc).__name__, _short(exc, 200))}
            (res["failures"] if is_check else res["statement_errors"]).append(entry)
sys.__stdout__.write("\n''' + _MARKER + r'''" + json.dumps(res) + "\n")
sys.__stdout__.flush()
'''


def _empty_result(tests: List[str], error: str, mode: str) -> Dict:
    return {
        "mode": mode, "loaded": False, "load_error": error, "n_asserts": 0, "n_ok": 0, "n_statements": 0,
        "converted": 0, "syntax_errors": 0, "failures": [], "statement_errors": [], "values": {},
        "defined_names": [], "referenced": False, "passed": False, "n_tests": len(tests),
    }


def run_harness(
    solution: str,
    tests: List[str],
    setup: str = "",
    timeout: float = 20.0,
    memory_mb: int = 2048,
    mode: str = "tests",
) -> Dict:
    """Chạy harness trên các dòng test (theo thứ tự, cùng một namespace).

    Kết quả có thêm 'passed' (chỉ True khi nạp được, có >= 1 assert, mọi assert và dòng lệnh đều chạy
    đúng, và test có gọi tới hàm/lớp của lời giải) và 'timed_out'.
    """
    tests = [t for t in (tests or []) if isinstance(t, str) and t.strip()]
    with tempfile.TemporaryDirectory(prefix="agent0_harness_") as tmp:
        harness = os.path.join(tmp, "harness.py")
        payload_path = os.path.join(tmp, "payload.json")
        with open(harness, "w", encoding="utf-8") as fh:
            fh.write(_HARNESS)
        with open(payload_path, "w", encoding="utf-8") as fh:
            json.dump({"solution": solution or "", "tests": tests, "setup": setup or "", "mode": mode}, fh)
        cmd = [sys.executable, "-I", harness, payload_path, str(int(timeout) + 1), str(int(memory_mb) * 1024 * 1024)]
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": tmp,
            "PYTHONHASHSEED": "0",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        try:
            proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            res = _empty_result(tests, f"Timeout after {timeout}s", mode)
            res["timed_out"] = True
            return res
    result: Optional[Dict] = None
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith(_MARKER):
            try:
                result = json.loads(line[len(_MARKER):])
            except json.JSONDecodeError:
                result = None
            break
    if result is None:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        return _empty_result(tests, f"Harness process ended abnormally (exit {proc.returncode}): {tail}", mode)
    result["timed_out"] = False
    result["n_tests"] = len(tests)
    result["passed"] = bool(
        result["loaded"]
        and result["n_asserts"] > 0
        and result["n_ok"] == result["n_asserts"]
        and not result["statement_errors"]
        and not result["syntax_errors"]
        and result["referenced"]
    )
    return result


def format_report(res: Dict) -> str:
    """Báo cáo ngắn (tiếng Anh, để đưa vào prompt chẩn đoán/sửa lỗi của LLM)."""
    lines: List[str] = []
    if res.get("timed_out"):
        return f"FAILED: {res.get('load_error') or 'timeout'} (infinite loop or too slow?)"
    if not res.get("loaded"):
        return f"FAILED to load the solution: {res.get('load_error')}"
    if res.get("n_asserts", 0) == 0:
        lines.append("FAILED: the tests contain no assert statement.")
    if not res.get("referenced", True):
        lines.append("FAILED: the tests do not call any function/class defined by the solution "
                     f"(defined: {', '.join(res.get('defined_names', [])[:8]) or 'nothing'}).")
    lines.append(f"{res.get('n_ok', 0)}/{res.get('n_asserts', 0)} assertions passed.")
    for f in res.get("failures", [])[:8]:
        lines.append(f"FAIL: {f.get('test')}")
        if "actual" in f or "expected" in f:
            lines.append(f"  -> {f.get('left')} returned {f.get('actual', f.get('actual_error'))}, "
                         f"expected {f.get('expected', f.get('expected_error'))}")
        else:
            lines.append(f"  -> {f.get('error')}")
    for f in res.get("statement_errors", [])[:4]:
        lines.append(f"ERROR in test statement: {f.get('test')} -> {f.get('error')}")
    if res.get("converted"):
        lines.append(f"(note: {res['converted']} bare comparison line(s) were treated as assert)")
    return "\n".join(lines)
