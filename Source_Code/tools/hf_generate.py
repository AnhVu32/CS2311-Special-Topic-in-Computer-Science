"""Sinh lời giải MBPP+ bằng transformers (model trên Hugging Face hoặc thư mục local, có thể kèm LoRA).

Dùng cho model không chạy qua API/Ollama, vd model sau GRPO (tools/grpo_executor.py). Prompt, cách chọn
few-shot và cách lấy code giống hệt run_benchmark, nên file kết quả chấm được bằng
`run_benchmark evaluate` và tổng hợp được bằng tools/summarize_matrix.py:

    python tools/hf_generate.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --output ../Data/results/hf/qwen2.5-coder-1.5b-hf/baseline_samples.jsonl
    python tools/hf_generate.py --model <thư mục model đã merge> --example_selection retrieved --n_examples 2 \\
        --kb_path ../Data/knowledge_base/all_models_clean/knowledge_base.jsonl --exclude_test_repaired \\
        --output ../Data/results/hf/qwen2.5-coder-1.5b-grpo/with_kb_n2_notr_samples.jsonl

Giải mã tham lam (temperature 0). Bài model không trả code vẫn được ghi (EvalPlus tính là sai), để lần chạy
luôn đủ 378 bài: giải mã tham lam chạy lại cũng cho đúng câu trả lời đó.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

SOURCE_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_CODE))

from agent0.scripts.run_benchmark import (  # noqa: E402
    DEFAULT_KB_PATH,
    build_messages,
    build_selector,
    environment_info,
    extract_solution,
    is_stub,
    kb_info,
    load_mbpp_plus_tasks,
    now_iso,
    read_jsonl,
    sidecar,
    write_json,
)

logger = logging.getLogger("hf_generate")
SELECTIONS = ("none", "retrieved", "random", "fixed")


def selector_args(args) -> argparse.Namespace:
    """Các trường build_selector của run_benchmark cần."""
    return argparse.Namespace(
        example_selection=args.example_selection, n_examples=args.n_examples, embed_model=args.embed_model,
        embed_base_url=args.embed_base_url, exclude_test_repaired=args.exclude_test_repaired,
        fewshot_format=args.fewshot_format, fewshot_asserts=1, min_similarity=args.min_similarity,
        mmr_lambda=1.0, retrieval_query="description", seed=args.seed,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="Tên model trên Hugging Face hoặc thư mục local")
    ap.add_argument("--adapter", default=None, help="Thư mục LoRA (gộp vào model trước khi sinh)")
    ap.add_argument("--output", required=True, help="<thư mục>/<tên lần chạy>_samples.jsonl")
    ap.add_argument("--example_selection", choices=SELECTIONS, default="none")
    ap.add_argument("--n_examples", type=int, default=2)
    ap.add_argument("--kb_path", default=str(DEFAULT_KB_PATH))
    ap.add_argument("--exclude_test_repaired", action="store_true")
    ap.add_argument("--fewshot_format", choices=("tests", "full", "code"), default="tests")
    ap.add_argument("--min_similarity", type=float, default=0.0)
    ap.add_argument("--embed_model", default="mxbai-embed-large")
    ap.add_argument("--embed_base_url", default="http://localhost:11434")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--chunk_tasks", type=int, default=64, help="Ghi kết quả sau mỗi chừng này bài")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--device", default=None, help="cuda / cuda:1 / cpu (mặc định: cuda nếu có)")
    ap.add_argument("--dtype", default="auto", help="auto (fp16 trên GPU, fp32 trên CPU) / float16 / float32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="Chỉ sinh N bài đầu (chạy thử)")
    ap.add_argument("--resume", action="store_true", help="Giữ các bài đã có trong --output, sinh phần còn thiếu")
    ap.add_argument("--log_level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="[%(asctime)s] %(levelname)-7s %(name)s - %(message)s",
                        datefmt="%H:%M:%S")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tasks = load_mbpp_plus_tasks()
    if args.limit:
        tasks = tasks[: args.limit]
    done = set()
    if out.exists():
        if not args.resume:
            raise SystemExit(f"{out} đã có. Thêm --resume để sinh tiếp, hoặc xoá file.")
        done = {r.get("task_id") for r in read_jsonl(out)}
    todo = [t for t in tasks if t["task_id"] not in done]
    if not todo:
        print(json.dumps({"output": str(out), "generated": 0, "already": len(done)}, ensure_ascii=False))
        return 0

    selector = None
    if args.example_selection != "none" and args.n_examples > 0:
        selector = build_selector(selector_args(args), tasks, Path(args.kb_path))
    infos, convs = [], []
    for task in todo:
        few_shot, info = selector.select(task) if selector is not None else ("", None)
        infos.append(info)
        convs.append(build_messages(task, few_shot, "evalplus"))

    from agent0.training.hf_backend import HFGenerator

    started = now_iso()
    t0 = time.time()
    gen = HFGenerator(args.model, adapter=args.adapter, device=args.device, dtype=args.dtype,
                      max_new_tokens=args.max_new_tokens)
    n_nocode = 0
    # Ghi sau mỗi phần (chunk_tasks bài): tiến trình bị dừng giữa chừng thì --resume chỉ sinh phần còn thiếu
    chunk = max(args.batch_size, args.chunk_tasks)
    fr = sidecar(out, "retrieval.jsonl").open("a", encoding="utf-8") if selector is not None else None
    try:
        with out.open("a", encoding="utf-8") as fh:
            for start in range(0, len(todo), chunk):
                part = slice(start, start + chunk)
                answers = gen.generate(convs[part], n=1, temperature=0.0, batch_size=args.batch_size,
                                       seed=args.seed, progress=False)
                for task, info, ans in zip(todo[part], infos[part], answers):
                    raw = ans[0] if ans else ""
                    solution = extract_solution(raw)
                    if not solution or is_stub(solution):
                        n_nocode += 1
                        solution = "# model không trả về code\n" + "\n".join(
                            f"# {line}" for line in (raw or "").splitlines()[:20])
                    fh.write(json.dumps({"task_id": task["task_id"], "solution": solution}, ensure_ascii=False) + "\n")
                    if fr is not None and info is not None:
                        fr.write(json.dumps({"task_id": task["task_id"], **info}, ensure_ascii=False) + "\n")
                fh.flush()
                if fr is not None:
                    fr.flush()
                logger.info("đã ghi %s/%s bài (%.0f s)", min(start + chunk, len(todo)), len(todo), time.time() - t0)
    finally:
        if fr is not None:
            fr.close()
    elapsed = round(time.time() - t0, 1)
    config = {
        "backend": "transformers", "model": args.model, "adapter": args.adapter, "device": gen.device,
        "dtype": str(gen.dtype), "prompt_style": "evalplus", "temperature": 0.0, "max_new_tokens": args.max_new_tokens,
        "mode": "with_kb" if selector is not None else "baseline",
        "n_examples": args.n_examples if selector is not None else 0, "limit": args.limit,
        "kb": kb_info(Path(args.kb_path)) if selector is not None else None,
        "fewshot": selector.describe() if selector is not None else None,
    }
    stats = {"generated": len(todo), "no_code": n_nocode, "seconds": elapsed}
    meta_path = sidecar(out, "meta.json")
    history = json.loads(meta_path.read_text(encoding="utf-8")).get("invocations", []) if meta_path.exists() else []
    entry = {"started_at": started, "finished_at": now_iso(), "command": sys.argv, "config": config, "stats": stats,
             "environment": environment_info()}
    write_json(meta_path, {"samples_file": str(out), "latest": entry, "invocations": history + [entry]})
    print(json.dumps({"output": str(out), **stats}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
