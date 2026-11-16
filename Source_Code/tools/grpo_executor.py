"""RL cho executor kiểu Agent0, thu nhỏ cho Kaggle (T4): GRPO + LoRA trên đề trong KB, thưởng = chạy test trong harness.

    prepare  đề KB -> prompt dạng MBPP+, khử ô nhiễm, chia train/val; đo p_hat của chính model (k mẫu/đề)
             và giữ đề ở biên năng lực, như bộ lọc |p_hat - 0.5| <= 0.25 của executor Agent0
    train    trl.GRPOTrainer + LoRA (fp16 trên T4), dừng theo số bước hoặc ngân sách thời gian, lưu adapter
    merge    gộp LoRA vào model gốc (để sinh bằng tools/hf_generate.py hoặc phục vụ bằng vLLM)
    score    pass@1 (giải mã tham lam, chạy đủ test) trên tập val — đo trong phân phối, trước / sau RL

Ví dụ (từ Source_Code):
    python tools/grpo_executor.py prepare --kb ../Data/knowledge_base/all_models_clean/knowledge_base.jsonl \\
        --out_dir /kaggle/working/grpo/data --model Qwen/Qwen2.5-Coder-1.5B-Instruct
    python tools/grpo_executor.py train --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --train /kaggle/working/grpo/data/train.jsonl --out_dir /kaggle/working/grpo/run --max_steps 200
    python tools/grpo_executor.py merge --model Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --adapter /kaggle/working/grpo/run/adapter --out /kaggle/working/grpo/merged
    python tools/grpo_executor.py score --model /kaggle/working/grpo/merged \\
        --tasks /kaggle/working/grpo/data/val.jsonl --out /kaggle/working/grpo/val_grpo.json

So với Agent0: executor học từ test đã kiểm chứng (Agent0 dùng nhãn giả đa số), một vòng (Agent0: 3),
model 1.5B + LoRA (Agent0: 4B/8B toàn bộ tham số), 8 rollout/đề (Agent0: 16), rollout một lượt.
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

SOURCE_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_CODE))

from agent0.training.data import (  # noqa: E402
    frontier_filter,
    load_rl_tasks,
    read_jsonl,
    split_train_val,
    write_jsonl,
)
from agent0.training.rewards import REWARD_MODES, HarnessReward, score_completion  # noqa: E402

logger = logging.getLogger("grpo_executor")
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def write_json(path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def score_rows(gen, rows, n=1, temperature=0.0, top_p=1.0, batch_size=16, max_new_tokens=512, seed=0, timeout=10.0):
    """Sinh n lời giải cho mỗi đề và chấm bằng harness. Trả về list (số lần qua hết test, n, tỉ lệ assert TB)."""
    answers = gen.generate([r["prompt"] for r in rows], n=n, temperature=temperature, top_p=top_p,
                           batch_size=batch_size, seed=seed, max_new_tokens=max_new_tokens)
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(i, a) for i, ans in enumerate(answers) for a in ans]
    with ThreadPoolExecutor(max_workers=8) as ex:
        scores = list(ex.map(lambda ja: (ja[0], score_completion(ja[1], rows[ja[0]]["tests"], rows[ja[0]]["setup"],
                                                                  timeout)), jobs))
    out = [[0, 0, 0.0] for _ in rows]
    for i, s in scores:
        out[i][0] += int(s["passed"])
        out[i][1] += 1
        out[i][2] += s["pass_frac"]
    return [(p, k, round(f / max(1, k), 4)) for p, k, f in out]


# ───────────────────────────── prepare ─────────────────────────────

def cmd_prepare(args) -> int:
    rows, stats = load_rl_tasks(args.kb, exclude_test_repaired=not args.keep_test_repaired,
                                decontam_threshold=args.decontam, min_asserts=args.min_asserts)
    train, val = split_train_val(rows, args.val_frac, args.seed)
    report = {"kb": [str(k) for k in args.kb], "load": stats, "n_train_pool": len(train), "n_val": len(val)}
    if args.model and args.p_hat_k > 0:
        from agent0.training.hf_backend import HFGenerator

        t0 = time.time()
        gen = HFGenerator(args.model, device=args.device, max_new_tokens=args.max_new_tokens)
        res = score_rows(gen, train, n=args.p_hat_k, temperature=args.temperature, top_p=args.top_p,
                         batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, seed=args.seed)
        p_hats = {}
        for row, (passes, k, frac) in zip(train, res):
            row["p_hat"] = round(passes / k, 4) if k else None
            row["pass_frac_mean"] = frac
            p_hats[row["uid"]] = row["p_hat"]
        hist = {}
        for p in p_hats.values():
            hist[f"{p:.2f}"] = hist.get(f"{p:.2f}", 0) + 1
        chosen, rule = frontier_filter(train, p_hats, tuple(args.band), args.min_train)
        report.update({"p_hat": {"model": args.model, "k": args.p_hat_k, "temperature": args.temperature,
                                 "histogram": dict(sorted(hist.items())), "seconds": round(time.time() - t0, 1)},
                       "frontier_rule": rule, "n_train": len(chosen)})
        write_jsonl(Path(args.out_dir) / "p_hat.jsonl",
                    [{"uid": r["uid"], "p_hat": r["p_hat"], "pass_frac_mean": r["pass_frac_mean"]} for r in train])
        train = chosen
    else:
        for row in train:
            row["p_hat"] = row.get("kb_p_hat")
        report.update({"frontier_rule": "không lọc (không đo p_hat)", "n_train": len(train)})
    write_jsonl(Path(args.out_dir) / "train.jsonl", train)
    write_jsonl(Path(args.out_dir) / "val.jsonl", val)
    write_json(Path(args.out_dir) / "prepare_report.json", report)
    print(json.dumps({k: report[k] for k in ("n_train_pool", "n_val", "n_train", "frontier_rule")}, ensure_ascii=False))
    if report.get("p_hat"):
        print("p_hat của model trên tập train:", report["p_hat"]["histogram"])
    return 0 if train else 2


# ───────────────────────────── train ─────────────────────────────

def cmd_train(args) -> int:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from agent0.training.hf_backend import load_tokenizer

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.train)
    if args.max_train:
        rows = rows[: args.max_train]
    if not rows:
        raise SystemExit(f"{args.train} rỗng")
    ds = Dataset.from_list([{"prompt": r["prompt"], "tests": r["tests"], "setup": r.get("setup") or "",
                             "p_hat": float(r["p_hat"]) if r.get("p_hat") is not None else float("nan"),
                             "uid": r["uid"]} for r in rows])

    cuda = torch.cuda.is_available()
    tok = load_tokenizer(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16 if cuda else torch.float32,
                                                 attn_implementation="sdpa")
    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                             target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    reward = HarnessReward(mode=args.reward, format_bonus=args.format_bonus, timeout=args.reward_timeout,
                           workers=args.reward_workers, adpo=args.adpo, s_min=args.s_min)
    vllm_kwargs = {}
    if args.use_vllm:
        vllm_kwargs = {"use_vllm": True, "vllm_mode": args.vllm_mode,
                       "vllm_gpu_memory_utilization": args.vllm_gpu_memory_utilization}
    config = GRPOConfig(
        output_dir=str(out_dir / "checkpoints"),
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        loss_type=args.loss_type,
        scale_rewards="none" if args.adpo else args.scale_rewards,
        mask_truncated_completions=args.mask_truncated,
        gradient_checkpointing=cuda and not args.no_gradient_checkpointing,
        fp16=cuda,
        bf16=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        disable_tqdm=True,
        seed=args.seed,
        dataloader_num_workers=0,
        **vllm_kwargs,
    )

    log_file = (out_dir / "train_log.jsonl").open("a", encoding="utf-8")
    deadline = time.time() + args.time_budget_minutes * 60 if args.time_budget_minutes else None
    t_start = time.time()

    class Progress(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            logs = dict(logs or {})
            logs["step"] = state.global_step
            logs["elapsed_min"] = round((time.time() - t_start) / 60, 2)
            if reward.history:
                logs["harness"] = reward.history[-1]
            log_file.write(json.dumps(logs, ensure_ascii=False, default=str) + "\n")
            log_file.flush()
            keys = [k for k in ("loss", "reward", "rewards/harness_reward/mean", "completions/mean_length",
                                "frac_reward_zero_std") if k in logs]
            msg = " | ".join(f"{k.split('/')[-2] if k.count('/') >= 2 else k}={logs[k]:.4g}"
                             for k in keys if isinstance(logs[k], (int, float)))
            if msg:
                h = reward.history[-1] if reward.history else {}
                print(f"[step {state.global_step}/{a.max_steps} | {logs['elapsed_min']} phút] {msg}"
                      + (f" | qua hết test={h.get('all_pass')}" if h else ""), flush=True)

        def on_step_end(self, a, state, control, **kw):
            if deadline and time.time() > deadline:
                print(f"Hết ngân sách {args.time_budget_minutes} phút ở bước {state.global_step} -> dừng và lưu.",
                      flush=True)
                control.should_training_stop = True
                control.should_save = True
            return control

    trainer = GRPOTrainer(model=model, reward_funcs=[reward], args=config, train_dataset=ds, processing_class=tok,
                          peft_config=peft_config, callbacks=[Progress()])
    from transformers.trainer_callback import PrinterCallback

    trainer.remove_callback(PrinterCallback)  # Progress đã in dòng gọn; bỏ dict log đầy đủ của Trainer
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"GRPO: {len(rows)} đề, {args.num_generations} rollout/đề, "
          f"{args.per_device_batch * args.grad_accum // args.num_generations} đề/bước, tối đa {args.max_steps} bước, "
          f"LoRA r={args.lora_r} ({trainable / 1e6:.2f}M tham số học), {'GPU fp16' if cuda else 'CPU fp32'}", flush=True)
    resume = None
    ckpts = sorted((out_dir / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    if args.resume and ckpts:
        resume = str(ckpts[-1])
        print(f"Chạy tiếp từ {resume}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume)
    adapter_dir = out_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    log_file.close()
    summary = {
        "model": args.model, "train_file": str(args.train), "n_train": len(rows),
        "steps": trainer.state.global_step, "minutes": round((time.time() - t_start) / 60, 2),
        "train_loss": getattr(result, "training_loss", None), "adapter": str(adapter_dir),
        "config": {k: v for k, v in vars(args).items() if k not in ("func",)},
        "reward_first": reward.history[: 5], "reward_last": reward.history[-5:],
    }
    write_json(out_dir / "train_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("steps", "minutes", "train_loss", "adapter")}, ensure_ascii=False))
    return 0


# ───────────────────────────── merge / score ─────────────────────────────

def cmd_merge(args) -> int:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from agent0.training.hf_backend import load_tokenizer

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    out = Path(args.out)
    merged.save_pretrained(str(out), safe_serialization=True)
    tok = load_tokenizer(args.model)
    tok.padding_side = "left"
    tok.save_pretrained(str(out))
    print(json.dumps({"merged": str(out), "dtype": args.dtype}, ensure_ascii=False))
    return 0


def cmd_score(args) -> int:
    from agent0.training.hf_backend import HFGenerator

    rows = read_jsonl(args.tasks)
    if args.limit:
        rows = rows[: args.limit]
    t0 = time.time()
    gen = HFGenerator(args.model, adapter=args.adapter, device=args.device, max_new_tokens=args.max_new_tokens)
    res = score_rows(gen, rows, n=1, temperature=0.0, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)
    per_task = [{"uid": r["uid"], "passed": bool(p), "pass_frac": f} for r, (p, k, f) in zip(rows, res)]
    out = {
        "model": args.model, "adapter": args.adapter, "tasks": str(args.tasks), "n": len(rows),
        "pass@1": round(100 * sum(t["passed"] for t in per_task) / max(1, len(rows)), 2),
        "mean_pass_frac": round(100 * sum(t["pass_frac"] for t in per_task) / max(1, len(rows)), 2),
        "seconds": round(time.time() - t0, 1), "per_task": per_task,
    }
    write_json(args.out, out)
    print(json.dumps({k: out[k] for k in ("n", "pass@1", "mean_pass_frac", "seconds")}, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log_level", default="INFO")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="KB -> train/val, đo p_hat và lọc đề ở biên năng lực")
    p.add_argument("--kb", nargs="+", required=True, help="Một hoặc nhiều knowledge_base.jsonl")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default=None, help="Model để đo p_hat (bỏ trống: không lọc theo p_hat)")
    p.add_argument("--p_hat_k", type=int, default=4)
    p.add_argument("--band", type=float, nargs=2, default=[0.25, 0.75], metavar=("LOW", "HIGH"))
    p.add_argument("--min_train", type=int, default=64, help="Ít đề hơn thì nới khoảng p_hat")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.99)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32, help="Số chuỗi sinh mỗi lô")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--decontam", type=float, default=0.5, help="Bỏ đề giống bài MBPP+ (cosine TF-IDF >= ngưỡng)")
    p.add_argument("--min_asserts", type=int, default=3)
    p.add_argument("--keep_test_repaired", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prepare)

    t = sub.add_parser("train", help="GRPO + LoRA")
    t.add_argument("--model", required=True)
    t.add_argument("--train", required=True, help="train.jsonl của bước prepare")
    t.add_argument("--out_dir", required=True)
    t.add_argument("--max_steps", type=int, default=200)
    t.add_argument("--time_budget_minutes", type=float, default=None, help="Hết giờ thì dừng ở cuối bước và lưu")
    t.add_argument("--max_train", type=int, default=None)
    t.add_argument("--lr", type=float, default=1e-5)
    t.add_argument("--warmup_steps", type=int, default=5)
    t.add_argument("--lora_r", type=int, default=16)
    t.add_argument("--lora_alpha", type=int, default=32)
    t.add_argument("--lora_dropout", type=float, default=0.05)
    t.add_argument("--num_generations", type=int, default=8, help="Số rollout mỗi đề (Agent0: 16)")
    t.add_argument("--per_device_batch", type=int, default=4, help="Số completion mỗi micro-batch")
    t.add_argument("--grad_accum", type=int, default=8)
    t.add_argument("--max_completion", type=int, default=512)
    t.add_argument("--temperature", type=float, default=1.0, help="Agent0: 1.0")
    t.add_argument("--top_p", type=float, default=0.99, help="Agent0: 0.99")
    t.add_argument("--beta", type=float, default=0.0, help="Hệ số KL (Agent0: 0.01; 0 = không cần model tham chiếu)")
    t.add_argument("--epsilon", type=float, default=0.2)
    t.add_argument("--epsilon_high", type=float, default=0.28, help="Clip-higher (DAPO)")
    t.add_argument("--loss_type", default="dapo")
    t.add_argument("--scale_rewards", default="group", choices=("group", "batch", "none"))
    t.add_argument("--mask_truncated", action="store_true", help="Không tính loss cho completion bị cắt cụt")
    t.add_argument("--reward", choices=REWARD_MODES, default="fraction")
    t.add_argument("--format_bonus", type=float, default=0.1)
    t.add_argument("--reward_timeout", type=float, default=10.0)
    t.add_argument("--reward_workers", type=int, default=8)
    t.add_argument("--adpo", action="store_true", help="Xấp xỉ ADPO: scale_rewards=none, thưởng nhân s(p_hat)")
    t.add_argument("--s_min", type=float, default=0.5)
    t.add_argument("--use_vllm", action="store_true")
    t.add_argument("--vllm_mode", default="colocate", choices=("colocate", "server"))
    t.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.3)
    t.add_argument("--no_gradient_checkpointing", action="store_true")
    t.add_argument("--save_steps", type=int, default=25)
    t.add_argument("--resume", action="store_true", help="Chạy tiếp từ checkpoint mới nhất trong out_dir")
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)

    m = sub.add_parser("merge", help="Gộp LoRA vào model gốc")
    m.add_argument("--model", required=True)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    m.set_defaults(func=cmd_merge)

    s = sub.add_parser("score", help="pass@1 trên tập val (giải mã tham lam)")
    s.add_argument("--model", required=True)
    s.add_argument("--adapter", default=None)
    s.add_argument("--tasks", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--batch_size", type=int, default=16)
    s.add_argument("--max_new_tokens", type=int, default=1024)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--device", default=None)
    s.set_defaults(func=cmd_score)

    args = ap.parse_args()
    logging.basicConfig(level=args.log_level, format="[%(asctime)s] %(levelname)-7s %(name)s - %(message)s",
                        datefmt="%H:%M:%S")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
