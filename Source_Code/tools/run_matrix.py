"""Chạy benchmark MBPP+ cho nhiều model × nhiều cách chọn ví dụ few-shot, rồi chấm EvalPlus.

Dùng trên Kaggle (xem kaggle/README.md) hoặc máy Linux. Chạy từ thư mục Source_Code:

    python tools/run_matrix.py --roster kaggle/roster.json --results_dir /kaggle/working/results \\
        --configs baseline retrieved random fixed --n_examples 2 \\
        --kb_path ../Data/knowledge_base/all_models_clean/knowledge_base.jsonl --time_budget_hours 10.5

Cấu hình (--configs):
  baseline   không có ví dụ
  retrieved  n ví dụ giống nhất trong KB (embedding)
  random     n ví dụ ngẫu nhiên cho mỗi bài
  fixed      cùng n ví dụ cho mọi bài
(retrieved so với random / fixed tách tác dụng "truy hồi" khỏi tác dụng "có ví dụ".)

KB:
  --exclude_test_repaired  bỏ các mục KB có test từng bị sửa cho khớp lời giải (tên lần chạy thêm _notr)
  --kb_tag TAG             nhãn KB (chữ/số/-), thêm _kbTAG vào tên lần chạy để nhiều KB cùng nằm trong một
                           results_dir (vd so KB lọc theo p_hat với KB đối chứng); baseline dùng chung

Cách chạy:
- Mỗi (model, cấu hình) là một lệnh `run_benchmark generate --resume`: bài đã sinh được giữ, chỉ sinh bài còn thiếu.
  Chạy lại lệnh này (hoặc phiên Kaggle mới với kết quả cũ) là chạy tiếp.
- Hai "làn" chạy song song: model chạy trên máy này (Ollama local, GPU) và model gọi qua API.
  --local_lanes 2 (máy hai GPU): model local chia hai làn, hai model chạy cùng lúc, mỗi model một GPU.
- Mỗi model được gọi thử một lần (local thì `ollama pull` trước); không gọi được thì bỏ qua và ghi lý do.
  Hết hạn mức của provider (ngày / tuần / credit) thì dừng các lần chạy còn lại của model đó.
- Hết --time_budget_hours: dừng lệnh đang chạy (bài đã sinh vẫn giữ), không bắt đầu lệnh mới.
- Chấm EvalPlus chạy nền trong lúc model khác đang sinh (--eval background); --eval skip để chấm sau.
- Trạng thái ghi liên tục vào <results_dir>/matrix_status.json; log từng lệnh ở <results_dir>/logs/.
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

SOURCE_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SOURCE_CODE))

from agent0.core.config import DEFAULT_KB_PATH, resolve_runtime  # noqa: E402
from agent0.core.llm_client import chat_completion  # noqa: E402
from agent0.scripts.run_benchmark import (  # noqa: E402
    check_samples,
    fewshot_variant,
    load_expected_task_ids,
    run_name,
    sidecar,
)

CONFIGS = ("baseline", "retrieved", "random", "fixed")
LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def is_local(spec: dict) -> bool:
    base = spec.get("base_url")
    if base:
        return (urlparse(base).hostname or "") in LOCAL_HOSTS
    return spec["provider"] == "ollama"


def load_roster(path, only: Optional[List[str]] = None, providers: Optional[List[str]] = None,
                include_disabled: bool = False) -> List[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    defaults = data.get("defaults", {})
    models, seen = [], set()
    for raw in data["models"]:
        spec = {**defaults, **raw}
        label = spec.get("label") or spec["model"].replace("/", "_").replace(":", "_")
        spec["label"] = label
        if label in seen:
            raise SystemExit(f"roster: label trùng: {label}")
        seen.add(label)
        if only:
            if label not in only:
                continue
        elif not spec.get("enabled", True) and not include_disabled:
            continue
        if providers and spec["provider"] not in providers:
            continue
        models.append(spec)
    missing = set(only or []) - seen
    if missing:
        raise SystemExit(f"roster không có label: {', '.join(sorted(missing))}")
    return models


class Status:
    """matrix_status.json — ghi lại sau mỗi thay đổi (an toàn khi nhiều luồng)."""

    def __init__(self, path: Path, meta: dict):
        self.path = path
        self.lock = threading.Lock()
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.data = {"models": old.get("models", {}), "runs": old.get("runs", {}), "sessions": old.get("sessions", [])}
        self.data["sessions"].append(meta)
        self.save()

    def set_model(self, label: str, **fields):
        with self.lock:
            self.data["models"].setdefault(label, {}).update(fields)
            self._save()

    def set_run(self, key: str, **fields):
        with self.lock:
            self.data["runs"].setdefault(key, {}).update(fields, updated_at=now())
            self._save()

    def save(self):
        with self.lock:
            self._save()

    def _save(self):
        self.data["updated_at"] = now()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)


class Matrix:
    def __init__(self, args, models: List[dict]):
        self.args = args
        self.models = models
        self.results = Path(args.results_dir).resolve()
        self.logs = self.results / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.deadline = time.time() + args.time_budget_hours * 3600
        self.expected = load_expected_task_ids()
        self.print_lock = threading.Lock()
        self.status = Status(self.results / "matrix_status.json", {
            "started_at": now(), "argv": sys.argv[1:], "time_budget_hours": args.time_budget_hours,
            "configs": args.configs, "n_examples": args.n_examples, "kb_path": str(args.kb_path),
            "kb_tag": args.kb_tag, "exclude_test_repaired": args.exclude_test_repaired,
        })
        self.eval_pool = ThreadPoolExecutor(max_workers=1) if args.eval == "background" else None
        self.eval_futures: List[Future] = []
        self.blocked: Dict[str, str] = {}
        for m in models:
            self.status.set_model(m["label"], provider=m["provider"], model=m["model"], tier=m.get("tier"),
                                  params_b=m.get("params_b"), local=is_local(m))

    # ── tiện ích ──
    def say(self, msg: str):
        with self.print_lock:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def remaining(self) -> float:
        return self.deadline - time.time()

    def run_paths(self, spec: dict, config: str):
        if config == "baseline":
            name = "baseline"
        else:
            ns = argparse.Namespace(
                example_selection=config, fewshot_format=self.args.fewshot_format, fewshot_asserts=1,
                min_similarity=self.args.min_similarity, mmr_lambda=1.0, retrieval_query="description",
                exclude_test_repaired=self.args.exclude_test_repaired,
            )
            name = run_name("with_kb", self.args.n_examples, None, "", fewshot_variant(ns))
            if self.args.kb_tag:
                name += f"_kb{self.args.kb_tag}"
        out = self.results / spec["label"] / f"{name}_samples.jsonl"
        return f"{spec['label']}/{name}", out

    def eval_done(self, samples: Path) -> Optional[dict]:
        metrics = sidecar(samples, "metrics.json")
        if metrics.exists() and samples.exists() and metrics.stat().st_mtime >= samples.stat().st_mtime:
            data = json.loads(metrics.read_text(encoding="utf-8"))
            if not data.get("partial"):
                return data
        return None

    def run_cmd(self, cmd: List[str], log_path: Path, prefix: str, timeout: Optional[float], show=None) -> Optional[int]:
        """Chạy lệnh, ghi toàn bộ log ra file, chỉ in dòng quan trọng. None = bị dừng vì hết thời gian."""
        show = show or self.interesting
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n$ {' '.join(cmd)}\n")
            log.flush()
            proc = subprocess.Popen(
                cmd, cwd=SOURCE_CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

            def pump():
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    if show(line):
                        self.say(f"{prefix} {line.rstrip()[-300:]}")

            reader = threading.Thread(target=pump, daemon=True)
            reader.start()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.say(f"{prefix} hết ngân sách thời gian -> dừng (các bài đã sinh vẫn được giữ)")
                for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                    try:
                        os.killpg(proc.pid, sig)
                        proc.wait(timeout=wait)
                        break
                    except (subprocess.TimeoutExpired, ProcessLookupError):
                        continue
                reader.join(timeout=5)
                return None
            reader.join(timeout=5)
            return proc.returncode

    @staticmethod
    def interesting(line: str) -> bool:
        if any(k in line for k in ("FAILED", "Dừng sớm", "ERROR", "Traceback", "Resume:", "pass@1", "=> ")):
            return True
        if "Generating " in line:
            try:
                done, total = line.split("Generating ", 1)[1].split(" ", 1)[0].split("/")
                return int(done) % 50 == 0 or done == total
            except (ValueError, IndexError):
                return False
        return False

    # ── kiểm tra model ──
    def preflight(self, spec: dict) -> Optional[str]:
        label = spec["label"]
        if is_local(spec) and spec["provider"] == "ollama" and spec.get("pull", True):
            self.say(f"[{label}] ollama pull {spec['model']} ...")
            try:
                proc = subprocess.run(["ollama", "pull", spec["model"]], capture_output=True, text=True,
                                      timeout=max(60, min(3600, self.remaining())))
            except FileNotFoundError:
                return "chưa cài Ollama (lệnh `ollama` không có)"
            except subprocess.TimeoutExpired:
                return "ollama pull quá lâu"
            if proc.returncode != 0:
                return f"ollama pull lỗi: {(proc.stderr or proc.stdout).strip()[-300:]}"
        try:
            api_key = os.environ.get(spec["api_key_env"]) if spec.get("api_key_env") else None
            if spec.get("api_key_env") and not api_key:
                return f"chưa đặt biến môi trường {spec['api_key_env']}"
            rt = resolve_runtime(spec["provider"], spec["model"], spec.get("base_url"), api_key)
            rt.extra_body = spec.get("extra_body")
            chat_completion(rt, [{"role": "user", "content": "Reply with the single word OK."}],
                            temperature=0.0, max_tokens=int(spec.get("preflight_max_tokens", 512)),
                            max_retries=1, timeout=300)
        except Exception as exc:  # noqa: BLE001 - mọi lỗi đều là "không dùng được model này"
            return f"{type(exc).__name__}: {exc}"[:500]
        return None

    # ── sinh + chấm một lần chạy ──
    def generate_cmd(self, spec: dict, config: str, out: Path) -> List[str]:
        a = self.args
        cmd = [
            sys.executable, "-m", "agent0.scripts.run_benchmark", "--log_level", "INFO", "generate",
            "--provider", spec["provider"], "--model", spec["model"], "--output", str(out), "--resume",
            "--n_examples", str(a.n_examples), "--prompt_style", a.prompt_style,
            "--temperature", str(spec.get("temperature", a.temperature)), "--seed", str(a.seed),
            "--max_tokens", str(spec.get("max_tokens", a.max_tokens)),
            "--workers", str(spec.get("workers", a.workers)),
            "--max_consecutive_errors", str(a.max_consecutive_errors),
        ]
        if spec.get("base_url"):
            cmd += ["--base_url", spec["base_url"]]
        if spec.get("api_key_env"):
            cmd += ["--api_key_env", spec["api_key_env"]]
        if spec.get("extra_body"):
            cmd += ["--extra_body", json.dumps(spec["extra_body"])]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if config == "baseline":
            cmd += ["--mode", "baseline"]
        else:
            cmd += [
                "--mode", "with_kb", "--kb_path", str(a.kb_path), "--example_selection", config,
                "--fewshot_format", a.fewshot_format, "--min_similarity", str(a.min_similarity),
                "--embed_model", a.embed_model, "--embed_base_url", a.embed_base_url,
            ]
            if a.exclude_test_repaired:
                cmd += ["--exclude_test_repaired"]
        return cmd

    def stopped_reason(self, out: Path) -> Optional[str]:
        meta = sidecar(out, "meta.json")
        if not meta.exists():
            return None
        latest = json.loads(meta.read_text(encoding="utf-8")).get("latest", {})
        return (latest.get("stats") or {}).get("stopped_early")

    def do_run(self, spec: dict, config: str) -> None:
        label = spec["label"]
        key, out = self.run_paths(spec, config)
        prefix = f"[{key}]"
        expected = self.expected[: self.args.limit] if self.args.limit else self.expected
        rep = check_samples(out, expected) if out.exists() else None
        if rep and rep["complete"]:
            metrics = self.eval_done(out)
            if metrics:
                self.status.set_run(key, model=label, config=config, samples=str(out), state="evaluated",
                                    n_done=len(expected), metrics=_short(metrics))
                return
            self.status.set_run(key, model=label, config=config, samples=str(out), state="generated",
                                n_done=len(expected), n_missing=0)
            self.queue_eval(key, out)
            return
        if label in self.blocked:
            self.status.set_run(key, model=label, config=config, state="skipped", reason=self.blocked[label])
            return

        for attempt in range(1 + self.args.retry_passes):
            left = self.remaining()
            if left < self.args.min_minutes_per_run * 60:
                self.status.set_run(key, model=label, config=config, samples=str(out), state="not_started",
                                    reason="hết ngân sách thời gian")
                return
            self.say(f"{prefix} bắt đầu sinh (lượt {attempt + 1}, còn {left / 3600:.1f} h)")
            out.parent.mkdir(parents=True, exist_ok=True)
            t0 = time.time()
            self.status.set_run(key, model=label, config=config, samples=str(out), state="generating")
            code = self.run_cmd(self.generate_cmd(spec, config, out), self.logs / f"{label}__{out.stem}.log",
                                prefix, timeout=left)
            rep = check_samples(out, expected) if out.exists() else {"complete": False, "missing": expected}
            n_missing = len(rep["missing"])
            fields = dict(model=label, config=config, samples=str(out), n_done=len(expected) - n_missing,
                          n_missing=n_missing, gen_seconds=round(time.time() - t0), returncode=code)
            if code is None:
                self.status.set_run(key, state="partial", reason="hết ngân sách thời gian", **fields)
                return
            if rep["complete"]:
                self.status.set_run(key, state="generated", **fields)
                self.say(f"{prefix} xong sinh ({fields['gen_seconds'] // 60} phút)")
                self.queue_eval(key, out)
                return
            reason = self.stopped_reason(out)
            if reason and ("hạn mức" in reason or "quota" in reason.lower()):
                self.blocked[label] = reason
                self.status.set_model(label, status="quota_exhausted", reason=reason)
                self.status.set_run(key, state="partial", reason=reason, **fields)
                self.say(f"{prefix} {reason} -> bỏ qua các lần chạy còn lại của {label} trong phiên này")
                return
            self.status.set_run(key, state="partial", reason=reason or f"còn {n_missing} bài lỗi", **fields)
            self.say(f"{prefix} còn {n_missing} bài chưa có lời giải" + (f" ({reason})" if reason else ""))
            if reason:  # dừng sớm vì lỗi liên tiếp: thử lại ngay thường cũng vậy
                return

    def queue_eval(self, key: str, out: Path) -> None:
        if self.args.eval == "skip" or self.args.limit:
            return
        if self.eval_pool is None:
            self.evaluate(key, out)
        else:
            self.eval_futures.append(self.eval_pool.submit(self.evaluate, key, out))

    def evaluate(self, key: str, out: Path) -> None:
        cmd = [sys.executable, "-m", "agent0.scripts.run_benchmark", "evaluate", "--samples_file", str(out),
               "--evaluator", "local"]
        if self.args.eval_parallel:
            cmd += ["--parallel", str(self.args.eval_parallel)]
        self.status.set_run(key, state="evaluating")
        t0 = time.time()
        code = self.run_cmd(cmd, self.logs / f"{key.replace('/', '__')}.eval.log", f"[{key} eval]", timeout=4 * 3600,
                            show=lambda line: any(k in line for k in ("Traceback", "ERROR", "Error", "lỗi")))
        metrics = self.eval_done(out)
        if code == 0 and metrics:
            self.status.set_run(key, state="evaluated", metrics=_short(metrics), eval_seconds=round(time.time() - t0))
            self.say(f"[{key}] base {metrics['base_pass@1']:.2f} / plus {metrics['plus_pass@1']:.2f}")
        else:
            self.status.set_run(key, state="eval_error", reason=f"evaluate exit {code} (xem logs)")

    # ── điều phối ──
    def lane(self, name: str, specs: List[dict]) -> None:
        ready: Dict[str, bool] = {}
        jobs = ([(c, s) for c in self.args.configs for s in specs] if self.args.order == "config"
                else [(c, s) for s in specs for c in self.args.configs])
        for config, spec in jobs:
            label = spec["label"]
            if self.remaining() < self.args.min_minutes_per_run * 60:
                self.say(f"[{name}] hết ngân sách thời gian, dừng làn")
                break
            key, out = self.run_paths(spec, config)
            done = out.exists() and check_samples(
                out, self.expected[: self.args.limit] if self.args.limit else self.expected)["complete"]
            if label not in ready and not done:
                reason = self.preflight(spec)
                ready[label] = reason is None
                if reason:
                    self.blocked[label] = reason
                    self.status.set_model(label, status="unavailable", reason=reason)
                    self.say(f"[{label}] KHÔNG dùng được: {reason}")
                else:
                    self.status.set_model(label, status="ok", reason=None)
            try:
                self.do_run(spec, config)
            except Exception as exc:  # noqa: BLE001 - một lần chạy hỏng không làm dừng cả ma trận
                self.status.set_run(key, model=label, config=config, state="error", reason=f"{type(exc).__name__}: {exc}"[:500])
                self.say(f"[{key}] lỗi: {exc}")

    def run(self) -> int:
        local = [m for m in self.models if is_local(m)]
        remote = [m for m in self.models if not is_local(m)]
        self.say(f"{len(self.models)} model ({len(local)} local, {len(remote)} qua API) × {len(self.args.configs)} cấu hình; "
                 f"ngân sách {self.args.time_budget_hours} h; kết quả ở {self.results}")
        if self.args.lanes == 2:
            groups = split_lanes(local, self.args.local_lanes)
            lanes = [(f"local{i + 1}" if len(groups) > 1 else "local", g) for i, g in enumerate(groups)]
            lanes.append(("api", remote))
            if len(groups) > 1:
                self.say("model local chia " + str(len(groups)) + " làn: "
                         + " | ".join(", ".join(m["label"] for m in g) for g in groups))
        else:
            lanes = [("all", local + remote)]
        threads = [threading.Thread(target=self.lane, args=lane, daemon=True) for lane in lanes if lane[1]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if self.eval_pool is not None:
            pending = [f for f in self.eval_futures if not f.done()]
            if pending:
                self.say(f"chờ {len(pending)} lần chấm còn lại (tối đa {self.args.eval_grace_minutes} phút)")
            grace_end = time.time() + self.args.eval_grace_minutes * 60
            for f in self.eval_futures:
                try:
                    f.result(timeout=max(1, grace_end - time.time()))
                except Exception:  # noqa: BLE001 - quá giờ / lỗi đã ghi vào status
                    pass
            self.eval_pool.shutdown(wait=False, cancel_futures=True)
        self.status.save()
        runs = self.status.data["runs"]
        states: Dict[str, int] = {}
        for r in runs.values():
            states[r.get("state", "?")] = states.get(r.get("state", "?"), 0) + 1
        self.say("xong: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())))
        return 0


def split_lanes(specs: List[dict], n: int) -> List[List[dict]]:
    """Chia model local thành n làn cân theo params_b (lớn trước, vào làn đang nhẹ nhất); giữ thứ tự roster trong làn.

    Mỗi làn chạy lần lượt các model của nó; với n = số GPU, Ollama đặt model của mỗi làn lên GPU còn trống nhiều nhất.
    """
    n = max(1, min(n, len(specs)))
    if n == 1:
        return [list(specs)] if specs else []
    order = {m["label"]: i for i, m in enumerate(specs)}
    lanes: List[List[dict]] = [[] for _ in range(n)]
    load = [0.0] * n
    for m in sorted(specs, key=lambda s: -float(s.get("params_b") or 1.0)):
        i = load.index(min(load))
        lanes[i].append(m)
        load[i] += float(m.get("params_b") or 1.0)
    return [sorted(g, key=lambda m: order[m["label"]]) for g in lanes if g]


def _short(metrics: dict) -> dict:
    return {k: metrics.get(k) for k in ("n_tasks", "base_pass@1", "plus_pass@1", "evaluated_at")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--roster", required=True, help="File JSON danh sách model (xem kaggle/roster.json)")
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=CONFIGS)
    ap.add_argument("--only", nargs="+", default=None, help="Chỉ chạy các label này (kể cả label đang tắt)")
    ap.add_argument("--providers", nargs="+", default=None, help="Chỉ chạy model của các provider này")
    ap.add_argument("--skip_local", action="store_true", help="Bỏ các model chạy local (vd phiên không có GPU)")
    ap.add_argument("--n_examples", type=int, default=2)
    ap.add_argument("--kb_path", default=str(DEFAULT_KB_PATH))
    ap.add_argument("--kb_tag", default=None, help="Nhãn KB (chữ/số/-) thêm vào tên lần chạy: _kb<TAG>")
    ap.add_argument("--exclude_test_repaired", action="store_true",
                    help="Bỏ mục KB có test từng bị sửa cho khớp lời giải (tên lần chạy thêm _notr)")
    ap.add_argument("--fewshot_format", choices=("tests", "full", "code"), default="tests")
    ap.add_argument("--min_similarity", type=float, default=0.0)
    ap.add_argument("--embed_model", default="mxbai-embed-large")
    ap.add_argument("--embed_base_url", default="http://localhost:11434")
    ap.add_argument("--prompt_style", default="evalplus", choices=("evalplus", "with_tests"))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=1800)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_consecutive_errors", type=int, default=10)
    ap.add_argument("--retry_passes", type=int, default=1, help="Số lượt --resume thêm khi còn bài lỗi lẻ tẻ")
    ap.add_argument("--limit", type=int, default=None, help="Chỉ N bài đầu (chạy thử; không chấm)")
    ap.add_argument("--order", choices=("config", "model"), default="config",
                    help="config: baseline của mọi model trước (mặc định); model: xong từng model một")
    ap.add_argument("--lanes", type=int, choices=(1, 2), default=2, help="2: model local và model API chạy song song")
    ap.add_argument("--local_lanes", type=int, default=1,
                    help="Số làn cho model local (= số GPU): mỗi làn một model, Ollama đặt mỗi model lên một GPU. "
                         "Cần OLLAMA_MAX_LOADED_MODELS >= số làn + 1 (thêm model embedding)")
    ap.add_argument("--time_budget_hours", type=float, default=10.5)
    ap.add_argument("--min_minutes_per_run", type=float, default=10.0)
    ap.add_argument("--eval", choices=("background", "inline", "skip"), default="background")
    ap.add_argument("--eval_parallel", type=int, default=None)
    ap.add_argument("--eval_grace_minutes", type=float, default=45.0)
    args = ap.parse_args()
    if args.kb_tag is not None and not re.fullmatch(r"[A-Za-z0-9-]+", args.kb_tag):
        raise SystemExit("--kb_tag chỉ gồm chữ, số và dấu '-' (dấu '_' dùng để tách các phần của tên lần chạy)")

    models = load_roster(args.roster, args.only, args.providers)
    if args.skip_local:
        models = [m for m in models if not is_local(m)]
    if not models:
        raise SystemExit("Không còn model nào để chạy (kiểm tra roster / --only / --providers / --skip_local)")
    if any(c != "baseline" for c in args.configs) and not Path(args.kb_path).exists():
        raise SystemExit(f"Không thấy KB: {args.kb_path}")
    return Matrix(args, models).run()


if __name__ == "__main__":
    sys.exit(main())
