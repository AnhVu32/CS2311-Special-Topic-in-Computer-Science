#!/usr/bin/env python3
"""Sinh knowledge base (KB) theo curriculum.

Nguồn đề:
  synthetic (mặc định)  LLM lập kế hoạch subtopic rồi sinh đề + test (kiểu curriculum agent của Agent0).
  mbpp                  đề lấy từ file MBPP gốc, BỎ các bài thuộc MBPP+ (trừ khi --include_mbpp_plus).

Với mỗi đề:
  bỏ qua nếu đã có trong KB / đã xử lý trong lần chạy / quá giống một đề MBPP+ / gần trùng một mục KB
  -> sinh lời giải -> harness kiểm chứng -> sửa code
  -> nếu vẫn sai: đồng thuận giữa các lời giải độc lập (mặc định) hoặc LLM phân xử (cách cũ)
  -> [tuỳ chọn] p_hat: k lời giải lấy mẫu, giữ đề có p_hat trong [0.25, 0.75] như Agent0
     (--p_hat_kb: lời giải lấy mẫu kèm ví dụ truy hồi từ KB, tức p_hat của "model + KB")
  -> KB (<output_dir>/<model>/knowledge_base.jsonl + all_models) hoặc rejected.jsonl.

Đồng tiến hoá không huấn luyện (thay cho RL của Agent0 khi không có GPU để cập nhật trọng số):
  vòng t: --feedback_from <summary vòng t-1> (planner sinh đề theo p_hat từng subtopic)
          --p_hat_kb <KB_t> (biên năng lực đo trên model + KB_t)  ->  KB_{t+1} = KB_t + đề mới ở biên.
"""
import argparse
import hashlib
import json
import logging
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Set, Tuple

from agent0.core.config import (
    DEFAULT_DATA_FILE,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PROVIDER,
    DEFAULT_STRATEGY,
    MBPP_PLUS_TASK_IDS_FILE,
    SUPPORTED_PROVIDERS,
    RuntimeConfig,
    describe_runtime,
    resolve_runtime,
)
from agent0.core.llm_client import call_stats, configure_cache
from agent0.core.logger import setup_logger
from agent0.core.utils import (
    api_key_from_args,
    append_jsonl,
    parse_extra_body,
    resolve_existing_path,
    sanitized_argv,
    write_json,
)
from agent0.curriculum.generator import generate_synthetic_records_with_llm
from agent0.curriculum.strategies import (
    build_explanation,
    infer_taxonomy,
    maybe_enrich_explanation_with_llm,
    mutate_prompt_text,
    order_all_strategies,
    order_diversity,
    order_easy_medium_hard,
    order_mutate,
)
from agent0.curriculum.uncertainty import (
    DEFAULT_BAND,
    band_verdict,
    estimate_pass_rate,
    histogram,
    kb_few_shot,
    load_feedback,
    summarize_by_group,
)
from agent0.data.mbpp_loader import load_mbpp_plus_ids, load_mbpp_records
from agent0.data.text_similarity import TfidfIndex, build_mbpp_plus_index, prompt_key
from agent0.execution.agent import TEST_REPAIR_MODES, execute_task

logger = logging.getLogger("agent0")

PIPELINE_VERSION = 2
DIFFICULTIES = ("easy", "medium", "hard")


# ───────────────────────── chọn seed ─────────────────────────

def select_seed_records(records: List[dict], strategy: str, items_per_strategy: int) -> List[dict]:
    prepared = []
    for rec in records:
        meta = infer_taxonomy(rec)
        planned = rec.get("subtopic_difficulty")
        if planned in DIFFICULTIES:
            # Đề tự sinh: giữ độ khó planner đã đặt cho subtopic (regex taxonomy chỉ là ước lượng thô).
            meta["difficulty_inferred"] = meta["difficulty"]
            meta["difficulty"] = planned
        prepared.append({**rec, **meta})

    if strategy == "easy_medium_hard":
        return order_easy_medium_hard(prepared, items_per_strategy)
    if strategy == "diversity":
        return order_diversity(prepared, items_per_strategy)
    if strategy == "mutate":
        return order_mutate(prepared, items_per_strategy)
    if strategy == "all":
        return order_all_strategies(prepared, items_per_strategy)
    raise ValueError(f"Unknown strategy: {strategy}")


def _mbpp_number(task_id) -> Optional[int]:
    try:
        return int(str(task_id).split("/")[-1])
    except (TypeError, ValueError):
        return None


def make_uid(strategy: str, task_id, model: str, prompt: str, solution: str) -> str:
    """uid duy nhất theo nội dung (bản cũ: strategy-task_id-thứ tự -> 485 dòng chỉ có 335 uid)."""
    digest = hashlib.sha1(f"{model}\n{prompt_key(prompt)}\n{solution.strip()}".encode("utf-8")).hexdigest()[:10]
    return f"{strategy}-{task_id}-{digest}"


# ───────────────────────── một đề ─────────────────────────

def curriculum_entry_from_seed(
    seed: dict,
    runtime: RuntimeConfig,
    opts: argparse.Namespace,
    band: Tuple[float, float],
) -> Tuple[dict, Optional[dict]]:
    """Giải + kiểm chứng (+ p_hat) một đề. Trả về (entry, reject_row hoặc None)."""
    taxonomy = seed.get("taxonomy", [])
    strategy = seed.get("curriculum_strategy", "all")
    stage = seed.get("curriculum_stage", seed.get("difficulty", "medium"))
    curriculum_prompt, mutation_notes = mutate_prompt_text(seed["prompt"], strategy, stage, taxonomy)

    exec_result = execute_task(
        runtime,
        seed,
        repair_rounds=opts.repair_rounds,
        verify_timeout=opts.verify_timeout,
        test_repair_mode=opts.test_repair,
        consensus_solutions=opts.consensus_solutions,
        with_reasoning=not opts.skip_reasoning,
    )
    final = exec_result.record
    solution_code = exec_result.solution_code
    verification = exec_result.verification
    accepted = bool(exec_result.accepted)
    tests_modified = (
        list(final.get("test_list") or []) != list(seed.get("test_list") or [])
        or list(final.get("challenge_test_list") or []) != list(seed.get("challenge_test_list") or [])
    )

    uncertainty = None
    band_result = None
    reject_stage = None if accepted else "verification"
    reason = None
    if accepted and opts.uncertainty_samples > 0:
        few_shot, kb_examples = "", None
        retriever = getattr(opts, "p_hat_retriever", None)
        if retriever is not None:  # p_hat của "model + KB_t" (đồng tiến hoá không huấn luyện)
            few_shot, kb_examples = kb_few_shot(retriever, final, opts.p_hat_n_examples)
        uncertainty = estimate_pass_rate(
            runtime, final, solution_code, opts.uncertainty_samples,
            temperature=opts.uncertainty_temperature, timeout=opts.verify_timeout,
            base_seed=2000 + 1000 * int(opts.seed), few_shot=few_shot,
        )
        if kb_examples is not None:
            uncertainty["kb_examples"] = kb_examples
        band_result = band_verdict(uncertainty["p_hat"], band)
        if opts.p_hat_action == "filter" and band_result in ("too_easy", "too_hard"):
            accepted = False
            reject_stage = "p_hat"
            reason = f"p_hat={uncertainty['p_hat']} ngoài khoảng [{band[0]}, {band[1]}] ({band_result})"

    explanation = ""
    if accepted:
        explain_rec = dict(final, reference_solution=solution_code)
        if opts.llm_explanation:
            explanation, mutation_notes = maybe_enrich_explanation_with_llm(
                runtime=runtime, record=explain_rec, curriculum_meta=seed,
                curriculum_prompt=curriculum_prompt, mutation_notes=mutation_notes,
            )
        else:
            explanation = build_explanation(explain_rec, seed)

    entry = {
        "uid": make_uid(strategy, seed.get("task_id"), opts.model_label, seed["prompt"], solution_code),
        "pipeline_version": PIPELINE_VERSION,
        "source_model": opts.model_label,
        "source_task_id": seed.get("task_id"),
        "source_file": seed.get("source_file"),
        "subtopic": seed.get("subtopic"),
        "strategy": strategy,
        "curriculum_stage": stage,
        "taxonomy": taxonomy,
        "difficulty_score": seed.get("difficulty_score"),
        "difficulty": seed.get("difficulty"),
        "primary_taxonomy": seed.get("primary_taxonomy"),
        "secondary_taxonomy": seed.get("secondary_taxonomy", []),
        "curriculum_prompt": curriculum_prompt,
        "original_prompt": seed["prompt"],
        "function_name": seed.get("function_name"),
        "solution_code": solution_code,
        "solution_source": exec_result.solution_source,
        "reasoning": exec_result.reasoning,
        "explanation": explanation,
        "mutation_notes": mutation_notes,
        "test_setup_code": final.get("test_setup_code", ""),
        "test_list": final.get("test_list", []),
        "challenge_test_list": final.get("challenge_test_list", []),
        "tests_modified": tests_modified,
        "original_test_list": seed.get("test_list", []) if tests_modified else None,
        "original_challenge_test_list": seed.get("challenge_test_list", []) if tests_modified else None,
        "verification": verification,
        "repair_rounds_used": exec_result.repair_rounds_used,
        "test_repair_rounds_used": exec_result.test_repair_rounds_used,
        "test_repair_method": exec_result.test_repair_method,
        "judge_verdict": exec_result.judge_verdict,
        "consensus": exec_result.consensus,
        "p_hat": uncertainty["p_hat"] if uncertainty else None,
        "r_unc": uncertainty["r_unc"] if uncertainty else None,
        "uncertainty": uncertainty,
        "difficulty_band": band_result,
        "accepted": accepted,
        "created_at": int(time.time()),
    }

    reject_entry = None
    if not accepted:
        if reason is None:
            reason = verification.get("stderr") or verification.get("stdout") or "verification failed"
            if exec_result.consensus and exec_result.consensus.get("reason"):
                reason = f"{reason}\n[consensus] {exec_result.consensus['reason']}"
        reject_entry = {
            "uid": entry["uid"],
            "pipeline_version": PIPELINE_VERSION,
            "source_model": opts.model_label,
            "source_task_id": seed.get("task_id"),
            "subtopic": seed.get("subtopic"),
            "strategy": strategy,
            "curriculum_stage": stage,
            "taxonomy": taxonomy,
            "curriculum_prompt": curriculum_prompt,
            "original_prompt": seed["prompt"],
            "reject_stage": reject_stage,
            "reason": reason,
            "verification": verification,
            "judge_verdict": exec_result.judge_verdict,
            "consensus": exec_result.consensus,
            "p_hat": entry["p_hat"],
            "uncertainty": uncertainty,
            # giữ lời giải của đề bị loại vì p_hat (đã kiểm chứng đúng) để có thể dùng lại
            "solution_code": solution_code if reject_stage == "p_hat" else None,
            "test_list": final.get("test_list", []) if reject_stage == "p_hat" else None,
            "created_at": entry["created_at"],
        }
    return entry, reject_entry


# ───────────────────────── tiện ích đọc KB ─────────────────────────

def _read_rows(path: Optional[Path]) -> List[dict]:
    rows: List[dict] = []
    if path is None or not path.exists():
        return rows
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Bỏ dòng hỏng %s:%s (%s)", path, line_no, exc)
    return rows


def _row_prompt(row: dict) -> str:
    return row.get("original_prompt") or row.get("curriculum_prompt") or ""


# ───────────────────────── CLI ─────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent0 cho code: sinh đề -> giải -> kiểm chứng -> (lọc p_hat) -> knowledge base"
    )
    src = parser.add_argument_group("nguồn đề")
    src.add_argument("--source", choices=["synthetic", "mbpp"], default=None,
                     help="synthetic: LLM tự sinh đề (mặc định). mbpp: đề từ --data_file, bỏ các bài MBPP+. "
                          "Không ghi --source mà có --data_file thì dùng mbpp.")
    src.add_argument("--synthetic_only", action="store_true", default=False,
                     help="Giữ để tương thích lệnh cũ, tương đương --source synthetic.")
    src.add_argument("--synthetic_count", type=int, default=60, help="Số đề cần sinh (nguồn synthetic).")
    src.add_argument("--synthetic_domain", type=str, default="coding",
                     help="Chủ đề lớn cho planner (vd coding, algorithms).")
    src.add_argument("--synthetic_generator", type=str, default="auto", choices=["auto", "llm", "template"],
                     help="Giữ để tương thích lệnh cũ (bản CS2230). Chỉ còn sinh đề bằng LLM; 'template' không còn hỗ trợ.")
    src.add_argument("--feedback_from", type=str, default=None,
                     help="summary.json của lần chạy trước (có --uncertainty_samples): gửi p_hat theo subtopic cho planner.")
    src.add_argument("--data_file", type=str, default=None,
                     help=f"File đề dạng MBPP (.jsonl/.json/.parquet) cho nguồn mbpp. Mặc định: {DEFAULT_DATA_FILE}")
    src.add_argument("--include_mbpp_plus", action="store_true", default=False,
                     help="Cho phép dùng cả các bài MBPP+ làm seed (KB sẽ chứa đề của tập test — chỉ để thử nghiệm).")
    src.add_argument("--prompt_key", type=str, default="text", help="Tên trường đề trong file dữ liệu.")
    src.add_argument("--code_key", type=str, default="code", help="Tên trường lời giải mẫu.")
    src.add_argument("--test_key", type=str, default="test_list", help="Tên trường test.")
    src.add_argument("--setup_key", type=str, default="test_setup_code", help="Tên trường code setup của test.")
    src.add_argument("--limit", type=int, default=None, help="Chỉ nạp N đề đầu của file dữ liệu.")

    rt = parser.add_argument_group("model")
    rt.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=SUPPORTED_PROVIDERS)
    rt.add_argument("--model", type=str, default=DEFAULT_MODEL)
    rt.add_argument("--base_url", type=str, default=None)
    rt.add_argument("--api_key", type=str, default=None)
    rt.add_argument("--api_key_env", type=str, default=None,
                    help="Đọc API key từ biến môi trường này (vd GEMINI_API_KEY cho --provider custom)")
    rt.add_argument("--extra_body", type=str, default=None,
                    help='JSON gộp vào body mỗi request, vd \'{"reasoning_effort": "low"}\' hoặc \'{"options": {"num_ctx": 8192}}\'')
    rt.add_argument("--cache_dir", type=str, default=None,
                    help="Cache phản hồi LLM trên đĩa (chạy lại cùng lệnh không tốn lời gọi).")

    cur = parser.add_argument_group("curriculum")
    cur.add_argument("--strategy", type=str, default=DEFAULT_STRATEGY,
                     choices=["easy_medium_hard", "diversity", "mutate", "all"])
    cur.add_argument("--items_per_strategy", type=int, default=10)
    cur.add_argument("--seed", type=int, default=42)

    ver = parser.add_argument_group("kiểm chứng")
    ver.add_argument("--repair_rounds", type=int, default=2, help="Số vòng sửa code tối đa.")
    ver.add_argument("--verify_timeout", type=int, default=20, help="Giây cho mỗi lần chạy test.")
    ver.add_argument("--test_repair", choices=TEST_REPAIR_MODES, default="consensus",
                     help="Khi code đã sửa vẫn fail: consensus (đồng thuận giữa lời giải độc lập, mặc định), "
                          "llm (cách cũ: LLM sửa assert cho khớp code), none (không bao giờ sửa test).")
    ver.add_argument("--consensus_solutions", type=int, default=2,
                     help="Số lời giải độc lập sinh thêm cho bước đồng thuận.")

    unc = parser.add_argument_group("độ khó kiểu Agent0 (p_hat)")
    unc.add_argument("--uncertainty_samples", type=int, default=0,
                     help="k lời giải lấy mẫu cho mỗi đề đã nhận để tính p_hat (0 = tắt; nên 4–8).")
    unc.add_argument("--uncertainty_temperature", type=float, default=0.8)
    unc.add_argument("--p_hat_band", type=float, nargs=2, metavar=("LOW", "HIGH"), default=list(DEFAULT_BAND),
                     help="Khoảng p_hat được giữ (mặc định 0.25 0.75 như Agent0).")
    unc.add_argument("--p_hat_action", choices=["filter", "tag"], default="filter",
                     help="filter: đề ngoài khoảng vào rejected.jsonl; tag: vẫn nhận, chỉ ghi p_hat "
                          "(để sau tách KB theo p_hat bằng tools/split_kb_by_phat.py).")
    unc.add_argument("--p_hat_kb", type=str, default=None,
                     help="Đo p_hat khi lời giải lấy mẫu có kèm ví dụ truy hồi từ KB này (executor = model + KB_t), "
                          "như benchmark dùng KB. Lặp: KB_{t+1} = KB_t + đề mới ở biên -> đồng tiến hoá không cập nhật "
                          "trọng số. Cần Ollama + --embed_model; bỏ các mục KB có test bị sửa.")
    unc.add_argument("--p_hat_n_examples", type=int, default=2, help="Số ví dụ KB kèm theo khi đo p_hat (--p_hat_kb).")

    flt = parser.add_argument_group("khử trùng / ô nhiễm")
    flt.add_argument("--decontam_threshold", type=float, default=0.5,
                     help="Bỏ đề có cosine TF-IDF với một đề MBPP+ >= ngưỡng (0 = tắt). Hiệu chỉnh: xem "
                          "agent0/data/text_similarity.py.")
    flt.add_argument("--near_dup_threshold", type=float, default=0.9,
                     help="Bỏ đề gần trùng một mục đã có trong KB (cosine TF-IDF >= ngưỡng; 0 = tắt).")
    flt.add_argument("--skip_previously_rejected", action="store_true", default=False,
                     help="Bỏ qua đề đã nằm trong rejected.jsonl của các lần chạy trước.")

    out = parser.add_argument_group("đầu ra")
    out.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                     help="KB ghi vào <output_dir>/<model>/ (nguồn mbpp: <output_dir>/mbpp_seeded/<model>/).")
    out.add_argument("--kb_path", type=str, default=None, help="Ghi đè đường dẫn KB của model.")
    out.add_argument("--rejected_path", type=str, default=None)
    out.add_argument("--summary_path", type=str, default=None)
    out.add_argument("--no_global_kb", action="store_true", default=False,
                     help="Không ghi thêm vào KB chung all_models/knowledge_base.jsonl.")
    out.add_argument("--update_index", action="store_true", default=False,
                     help="Cuối lần chạy dựng lại index embedding của KB (cần Ollama + model embedding).")
    out.add_argument("--embed_model", type=str, default="mxbai-embed-large")
    out.add_argument("--embed_base_url", type=str, default="http://localhost:11434")
    out.add_argument("--max_output_examples", type=int, default=0, help="Không còn dùng; giữ để tương thích.")

    cost = parser.add_argument_group("chi phí")
    cost.add_argument("--llm_explanation", action="store_true", default=False,
                      help="Sinh trường explanation bằng LLM (+1 lời gọi/đề). Mặc định dựng từ template.")
    cost.add_argument("--skip_reasoning", action="store_true", default=False,
                      help="Không sinh trường reasoning (-1 lời gọi/đề nhận; định dạng few-shot 'full' dùng trường này).")
    cost.add_argument("--max_consecutive_errors", type=int, default=5,
                      help="Dừng sớm khi chừng này đề liên tiếp lỗi (vd hết quota API).")
    cost.add_argument("--workers", type=int, default=1,
                      help="Giải nhiều đề song song (vd 4; coi chừng giới hạn tốc độ của API).")

    log = parser.add_argument_group("log")
    log.add_argument("--log_file", type=str, default=None, help="Đường dẫn file log (vd ./logs/run.log)")
    log.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    log.add_argument("--log_llm_io", action="store_true", default=False, help="Ghi toàn bộ input/output LLM vào log")
    log.add_argument("--llm_io_max_chars", type=int, default=20000,
                     help="Số ký tự tối đa mỗi input/output LLM trong log (0 = không giới hạn)")
    log.add_argument("--quiet", action="store_true", default=False)
    return parser


def resolve_source(args) -> str:
    if args.synthetic_only:
        if args.source == "mbpp":
            raise SystemExit("--synthetic_only mâu thuẫn với --source mbpp")
        return "synthetic"
    if args.source:
        return args.source
    return "mbpp" if args.data_file else "synthetic"


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    started = time.time()

    setup_logger(
        log_file=args.log_file,
        log_level=args.log_level,
        quiet=args.quiet,
        log_llm_io=args.log_llm_io,
        llm_io_max_chars=args.llm_io_max_chars,
    )
    logger.info("=" * 80)
    logger.info("Agent0 code curriculum - start")
    logger.info("=" * 80)

    source = resolve_source(args)
    if source == "synthetic" and args.synthetic_generator == "template":
        raise SystemExit(
            "--synthetic_generator template chỉ có ở bản script cũ (CS2230). Bản này sinh đề bằng LLM: "
            "bỏ cờ này hoặc dùng --synthetic_generator llm."
        )
    low, high = args.p_hat_band
    if not 0.0 <= low <= high <= 1.0:
        raise SystemExit("--p_hat_band cần 0 <= LOW <= HIGH <= 1")
    band = (low, high)

    runtime = resolve_runtime(args.provider, args.model, args.base_url, api_key_from_args(args))
    runtime.extra_body = parse_extra_body(args.extra_body)
    logger.info("Runtime: %s | nguồn đề: %s", describe_runtime(runtime), source)
    # offline: không gọi LLM — dùng lời giải mẫu của bộ dữ liệu (chỉ với nguồn mbpp), KB ghi dưới tên "reference".
    model_label = runtime.model
    if runtime.provider == "offline":
        if source != "mbpp":
            raise SystemExit("Provider 'offline' không sinh được đề / lời giải. Dùng --source mbpp hoặc chọn provider khác.")
        if args.uncertainty_samples > 0:
            raise SystemExit("--uncertainty_samples cần gọi LLM, không dùng được với provider 'offline'.")
        model_label = "reference"
        logger.warning("Provider offline: KB dựng từ lời giải mẫu của %s, ghi dưới tên 'reference'.", args.data_file or DEFAULT_DATA_FILE)
    args.model_label = model_label
    configure_cache(args.cache_dir)

    # executor vòng t = model + KB_t: dựng index một lần trước khi chạy song song (truy vấn chỉ đọc)
    args.p_hat_retriever = None
    p_hat_kb_info = None
    if args.p_hat_kb:
        if args.uncertainty_samples <= 0:
            raise SystemExit("--p_hat_kb cần --uncertainty_samples > 0")
        from agent0.retrieval.retriever import KnowledgeRetriever

        p_hat_kb = resolve_existing_path(args.p_hat_kb)
        if not Path(p_hat_kb).exists():
            raise SystemExit(f"Không thấy KB cho --p_hat_kb: {p_hat_kb}")
        args.p_hat_retriever = KnowledgeRetriever(
            str(p_hat_kb), embed_model=args.embed_model, embed_base_url=args.embed_base_url, exclude_test_repaired=True,
        )
        args.p_hat_retriever.build_index()
        p_hat_kb_info = {"path": str(p_hat_kb), "n_examples": args.p_hat_n_examples, "entries": len(args.p_hat_retriever.entries),
                         "filter": dict(args.p_hat_retriever.stats)}
        logger.info("p_hat đo với KB %s (%s mục, %s ví dụ/đề)", p_hat_kb, p_hat_kb_info["entries"], args.p_hat_n_examples)

    root = resolve_existing_path(args.output_dir)
    if source == "mbpp":
        root = root / "mbpp_seeded"
    model_dir = root / model_label.replace("/", "_").replace(":", "_")
    kb_path = Path(args.kb_path) if args.kb_path else model_dir / "knowledge_base.jsonl"
    rejected_path = Path(args.rejected_path) if args.rejected_path else model_dir / "rejected.jsonl"
    summary_path = Path(args.summary_path) if args.summary_path else model_dir / "summary.json"
    global_kb_path = None if args.no_global_kb else root / "all_models" / "knowledge_base.jsonl"

    # ── nạp đề ──
    excluded_plus = 0
    data_path = None
    plus_ids: Set[int] = set()
    if MBPP_PLUS_TASK_IDS_FILE.exists():
        plus_ids = load_mbpp_plus_ids()
    if source == "synthetic":
        feedback = load_feedback(args.feedback_from) if args.feedback_from else None
        records = generate_synthetic_records_with_llm(
            runtime=runtime, count=args.synthetic_count, seed=args.seed, domain=args.synthetic_domain,
            feedback=feedback,
        )
    else:
        data_path = resolve_existing_path(args.data_file or str(DEFAULT_DATA_FILE))
        records = load_mbpp_records(
            data_file=str(data_path), prompt_key=args.prompt_key, code_key=args.code_key,
            test_key=args.test_key, setup_key=args.setup_key, limit=args.limit,
        )
        if not args.include_mbpp_plus:
            if not plus_ids:
                raise SystemExit(f"Thiếu {MBPP_PLUS_TASK_IDS_FILE}: không loại được các bài MBPP+ khỏi seed.")
            before = len(records)
            records = [r for r in records if _mbpp_number(r.get("task_id")) not in plus_ids]
            excluded_plus = before - len(records)
            logger.info("Loại %s bài thuộc MBPP+ khỏi seed (còn %s)", excluded_plus, len(records))
    seeds_loaded = len(records)

    selected = select_seed_records(records, args.strategy, args.items_per_strategy)
    if not selected:
        raise ValueError("No curriculum seeds selected. Check strategy/limit/items_per_strategy.")
    if source == "synthetic" and len(selected) < len(records):
        logger.warning("Chỉ giải %s/%s đề đã sinh (--strategy %s, --items_per_strategy %s); đặt --items_per_strategy >= "
                       "--synthetic_count để không bỏ phí đề.", len(selected), len(records), args.strategy, args.items_per_strategy)

    # ── trạng thái KB hiện có ──
    existing_rows = _read_rows(kb_path)
    existing_keys = {prompt_key(_row_prompt(r)) for r in existing_rows}
    global_keys = {(r.get("source_model"), prompt_key(_row_prompt(r))) for r in _read_rows(global_kb_path)}
    rejected_keys = (
        {prompt_key(_row_prompt(r)) for r in _read_rows(rejected_path)} if args.skip_previously_rejected else set()
    )

    decontam = None
    if args.decontam_threshold > 0 and not args.include_mbpp_plus and plus_ids and DEFAULT_DATA_FILE.exists():
        decontam = build_mbpp_plus_index(DEFAULT_DATA_FILE, plus_ids)
    near_dup = None
    if args.near_dup_threshold > 0:
        fit_extra = []
        if DEFAULT_DATA_FILE.exists():
            fit_extra = [r.get("text", "") for r in _read_rows(DEFAULT_DATA_FILE)]
        kb_prompts = [_row_prompt(r) for r in existing_rows]
        near_dup = TfidfIndex(kb_prompts, ids=[r.get("uid") for r in existing_rows], fit_docs=fit_extra + kb_prompts)

    stats: Counter = Counter()
    taxonomy_counter: Counter = Counter()
    consensus_outcomes: Counter = Counter()
    judge_verdicts: Counter = Counter()
    solution_sources: Counter = Counter()
    reject_stages: Counter = Counter()
    unc_rows: List[dict] = []
    accepted_rows: List[dict] = []
    processed_keys: Set[str] = set()
    consecutive_errors = 0
    stopped_early = False
    interrupted = False

    def passes_filters(seed: dict) -> bool:
        """Các bước lọc rẻ (không gọi LLM) trước khi giải một đề."""
        key = prompt_key(seed.get("prompt", ""))
        if key in processed_keys:
            stats["skipped_duplicate_seed"] += 1
            return False
        processed_keys.add(key)
        if key in existing_keys:
            stats["skipped_existing_kb"] += 1
            return False
        if key in rejected_keys:
            stats["skipped_previously_rejected"] += 1
            return False
        if decontam is not None:
            sim, plus_id, _ = decontam.most_similar(seed["prompt"])
            if sim >= args.decontam_threshold:
                stats["skipped_contaminated"] += 1
                reject_stages["decontamination"] += 1
                logger.info("Bỏ đề giống %s (cosine %.2f) | task_id=%s", plus_id, sim, seed.get("task_id"))
                append_jsonl(rejected_path, {
                    "pipeline_version": PIPELINE_VERSION, "source_model": model_label,
                    "source_task_id": seed.get("task_id"), "subtopic": seed.get("subtopic"),
                    "original_prompt": seed["prompt"], "reject_stage": "decontamination",
                    "reason": f"cosine TF-IDF {sim:.3f} với {plus_id} >= {args.decontam_threshold}",
                    "created_at": int(time.time()),
                })
                return False
        if near_dup is not None and len(near_dup):
            sim, uid, _ = near_dup.most_similar(seed["prompt"])
            if sim >= args.near_dup_threshold:
                stats["skipped_near_duplicate"] += 1
                logger.info("Bỏ đề gần trùng mục KB %s (cosine %.2f) | task_id=%s", uid, sim, seed.get("task_id"))
                return False
        return True

    def handle_error(seed: dict, exc: BaseException) -> bool:
        """Ghi đề lỗi vào rejected; trả về True nếu phải dừng sớm."""
        nonlocal consecutive_errors
        consecutive_errors += 1
        stats["errors"] += 1
        reject_stages["exception"] += 1
        logger.error("Đề lỗi | task_id=%s | %s: %s", seed.get("task_id"), type(exc).__name__, str(exc)[:500])
        append_jsonl(rejected_path, {
            "pipeline_version": PIPELINE_VERSION, "source_model": model_label,
            "source_task_id": seed.get("task_id"), "subtopic": seed.get("subtopic"),
            "original_prompt": seed["prompt"], "reject_stage": "exception",
            "reason": f"{type(exc).__name__}: {exc}"[:2000], "created_at": int(time.time()),
        })
        if args.max_consecutive_errors and consecutive_errors >= args.max_consecutive_errors:
            logger.error("%s đề liên tiếp lỗi -> dừng (kiểm tra API key / quota / mạng).", consecutive_errors)
            return True
        return False

    def handle_result(seed: dict, entry: dict, rejected: Optional[dict]) -> None:
        nonlocal consecutive_errors
        consecutive_errors = 0
        key = prompt_key(seed.get("prompt", ""))
        taxonomy_counter.update(entry.get("taxonomy", []))
        stats[f"stage_{entry['curriculum_stage']}"] += 1
        stats[f"strategy_{entry['strategy']}"] += 1
        stats["verified_pass" if entry["verification"].get("passed") else "verified_fail"] += 1
        if entry["tests_modified"]:
            stats["tests_modified"] += 1
        if entry.get("consensus"):
            consensus_outcomes[entry["consensus"].get("outcome", "?")] += 1
        if entry.get("judge_verdict"):
            judge_verdicts[entry["judge_verdict"]] += 1
        if entry["verification"].get("passed"):
            solution_sources[entry.get("solution_source") or "model"] += 1
        if entry.get("uncertainty") is not None:
            unc_rows.append({
                "group": seed.get("subtopic") or seed.get("primary_taxonomy"),
                "difficulty": seed.get("subtopic_difficulty") or seed.get("difficulty"),
                "p_hat": entry["p_hat"], "r_unc": entry["r_unc"], "band": entry["difficulty_band"],
            })
        if entry["accepted"]:
            stats["accepted"] += 1
            accepted_rows.append(entry)
            append_jsonl(kb_path, entry)
            existing_keys.add(key)
            if global_kb_path is not None and (model_label, key) not in global_keys:
                append_jsonl(global_kb_path, entry)
                global_keys.add((model_label, key))
            if near_dup is not None:
                near_dup.add(seed["prompt"], entry["uid"])
        else:
            stats["rejected"] += 1
            if rejected is not None:
                reject_stages[rejected.get("reject_stage") or "verification"] += 1
                append_jsonl(rejected_path, rejected)

    try:
        if args.workers <= 1:
            for idx, seed in enumerate(selected, start=1):
                if not passes_filters(seed):
                    continue
                logger.info("Đề %s/%s | task_id=%s | %s", idx, len(selected), seed.get("task_id"), seed["prompt"][:80])
                stats["processed"] += 1
                try:
                    entry, rejected = curriculum_entry_from_seed(seed, runtime, args, band)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - một đề lỗi không được làm hỏng cả lần chạy
                    if handle_error(seed, exc):
                        stopped_early = True
                        break
                    continue
                handle_result(seed, entry, rejected)
        else:
            # Song song: lọc trước (tuần tự), giải nhiều đề cùng lúc; ghi KB ở luồng chính theo thứ tự xong.
            # Khử gần-trùng chỉ so với KB có sẵn + các đề đã xong trước lúc nộp.
            todo = [seed for seed in selected if passes_filters(seed)]
            stats["processed"] += len(todo)
            logger.info("Giải %s đề với %s luồng", len(todo), args.workers)
            pool = ThreadPoolExecutor(max_workers=args.workers)
            try:
                futures = {pool.submit(curriculum_entry_from_seed, seed, runtime, args, band): seed for seed in todo}
                for fut in as_completed(futures):
                    seed = futures[fut]
                    try:
                        entry, rejected = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        if handle_error(seed, exc):
                            stopped_early = True
                            pool.shutdown(wait=False, cancel_futures=True)
                            break
                        continue
                    handle_result(seed, entry, rejected)
            except KeyboardInterrupt:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            pool.shutdown(wait=not stopped_early)
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Bị dừng (Ctrl+C) — vẫn ghi summary cho phần đã chạy.")

    index_info = None
    if args.update_index and accepted_rows:
        try:
            from agent0.retrieval.retriever import KnowledgeRetriever

            retriever = KnowledgeRetriever(str(kb_path), embed_model=args.embed_model, embed_base_url=args.embed_base_url)
            retriever.build_index()
            index_info = dict(retriever.stats)
        except Exception as exc:  # noqa: BLE001 - index chỉ là cache, benchmark sẽ tự dựng lại
            logger.warning("Không dựng được index embedding (%s) — run_benchmark sẽ tự dựng khi cần.", exc)
            index_info = {"error": str(exc)}

    ps = [r["p_hat"] for r in unc_rows if r["p_hat"] is not None]
    uncertainty_summary = None
    if args.uncertainty_samples > 0:
        bands = Counter(r["band"] for r in unc_rows)
        uncertainty_summary = {
            "k": args.uncertainty_samples,
            "temperature": args.uncertainty_temperature,
            "band": list(band),
            "action": args.p_hat_action,
            "with_kb": bool(args.p_hat_kb),
            "n_estimated": len(unc_rows),
            "p_hat_mean": round(sum(ps) / len(ps), 3) if ps else None,
            "histogram": histogram(ps, args.uncertainty_samples),
            "in_band": bands.get("in_band", 0),
            "too_easy": bands.get("too_easy", 0),
            "too_hard": bands.get("too_hard", 0),
            "unknown": bands.get("unknown", 0),
        }

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started)),
        "elapsed_sec": round(time.time() - started, 1),
        "command": sanitized_argv(),
        "pipeline_version": PIPELINE_VERSION,
        "config": {
            "provider": runtime.provider,
            "backend": runtime.backend,
            "model": runtime.model,
            "model_label": model_label,
            "base_url": runtime.base_url,
            "extra_body": runtime.extra_body,
            "source": source,
            "data_file": str(data_path) if data_path else None,
            "include_mbpp_plus": args.include_mbpp_plus,
            "synthetic_count": args.synthetic_count if source == "synthetic" else None,
            "synthetic_domain": args.synthetic_domain if source == "synthetic" else None,
            "feedback_from": args.feedback_from,
            "strategy": args.strategy,
            "items_per_strategy": args.items_per_strategy,
            "limit": args.limit,
            "seed": args.seed,
            "repair_rounds": args.repair_rounds,
            "verify_timeout": args.verify_timeout,
            "test_repair": args.test_repair,
            "consensus_solutions": args.consensus_solutions,
            "uncertainty_samples": args.uncertainty_samples,
            "uncertainty_temperature": args.uncertainty_temperature,
            "p_hat_band": list(band),
            "p_hat_action": args.p_hat_action,
            "p_hat_kb": p_hat_kb_info,
            "decontam_threshold": args.decontam_threshold if decontam is not None else 0,
            "near_dup_threshold": args.near_dup_threshold,
            "llm_explanation": args.llm_explanation,
            "skip_reasoning": args.skip_reasoning,
            "cache_dir": args.cache_dir,
            "workers": args.workers,
        },
        "paths": {
            "kb": str(kb_path), "rejected": str(rejected_path), "summary": str(summary_path),
            "global_kb": str(global_kb_path) if global_kb_path else None,
        },
        "stats": {
            "seeds_loaded": seeds_loaded,
            "excluded_mbpp_plus": excluded_plus,
            "selected": len(selected),
            "processed": stats.get("processed", 0),
            "accepted": stats.get("accepted", 0),
            "rejected": stats.get("rejected", 0),
            "errors": stats.get("errors", 0),
            "skipped_existing_kb": stats.get("skipped_existing_kb", 0),
            "skipped_duplicate_seed": stats.get("skipped_duplicate_seed", 0),
            "skipped_previously_rejected": stats.get("skipped_previously_rejected", 0),
            "skipped_contaminated": stats.get("skipped_contaminated", 0),
            "skipped_near_duplicate": stats.get("skipped_near_duplicate", 0),
            "stopped_early": stopped_early,
            "interrupted": interrupted,
            "kb_existing": len(existing_rows),
            "kb_total": len(existing_rows) + len(accepted_rows),
            "tests_modified": stats.get("tests_modified", 0),
            "consensus_outcomes": dict(consensus_outcomes),
            "judge_verdicts": dict(judge_verdicts),
            "solution_sources": dict(solution_sources),
            "reject_stages": dict(reject_stages),
            "taxonomy_top": taxonomy_counter.most_common(15),
            "counts": dict(stats),
            "llm_calls": call_stats(),
        },
        "uncertainty": uncertainty_summary,
        "uncertainty_by_subtopic": summarize_by_group(unc_rows, band) if unc_rows else [],
        "index": index_info,
    }
    write_json(summary_path, summary)
    append_jsonl(summary_path.with_name(summary_path.stem + "_history.jsonl"), summary)
    logger.info(
        "Xong: nhận %s, loại %s, lỗi %s, bỏ qua (đã có %s, trùng %s, giống MBPP+ %s, gần trùng %s) | %s lời gọi LLM (%s từ cache)",
        summary["stats"]["accepted"], summary["stats"]["rejected"], summary["stats"]["errors"],
        summary["stats"]["skipped_existing_kb"], summary["stats"]["skipped_duplicate_seed"],
        summary["stats"]["skipped_contaminated"], summary["stats"]["skipped_near_duplicate"],
        summary["stats"]["llm_calls"]["calls"], summary["stats"]["llm_calls"]["cache_hits"],
    )
    logger.info("KB: %s | summary: %s", kb_path, summary_path)
    if interrupted:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
