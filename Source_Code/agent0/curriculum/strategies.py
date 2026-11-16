import json
import re
import textwrap
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from agent0.core.config import DIFFICULTY_WEIGHTS, TAXONOMY_RULES, RuntimeConfig
from agent0.core.llm_client import chat_completion
from agent0.core.utils import compact_sentence, parse_json_object, safe_strip_text

def infer_taxonomy(record: dict) -> dict:
    text = "\n".join(
        [
            record.get("prompt", ""),
            record.get("reference_solution", ""),
            "\n".join(record.get("test_list", [])),
            "\n".join(record.get("challenge_test_list", [])),
        ]
    ).lower()

    counts: Dict[str, int] = {}
    for label, patterns in TAXONOMY_RULES.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                score += 1
        if score:
            counts[label] = score

    code = record.get("reference_solution", "")
    code_features = {
        "for_loops": len(re.findall(r"\bfor\b", code)),
        "while_loops": len(re.findall(r"\bwhile\b", code)),
        "ifs": len(re.findall(r"\bif\b", code)),
        "recursion": bool(re.search(r"\breturn\s+\w+\(", code)) and any(k in counts for k in ["recursion", "dp", "graph"]),
        "comprehensions": len(re.findall(r"\[[^\]]+for[^\]]+\]", code)),
        "imports": len(re.findall(r"^\s*(?:from\s+\S+\s+import|import\s+\S+)", code, flags=re.M)),
    }

    primary = max(counts.items(), key=lambda kv: (kv[1], DIFFICULTY_WEIGHTS.get(kv[0], 0.05)))[0] if counts else "general"
    secondary = [label for label, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if label != primary][:4]

    prompt_len = len(record.get("prompt", ""))
    code_len = len(code)
    signals = len(counts)
    difficulty_score = 0.15
    difficulty_score += min(prompt_len / 600.0, 0.20)
    difficulty_score += min(code_len / 1200.0, 0.15)
    difficulty_score += min(signals * 0.08, 0.30)
    difficulty_score += sum(DIFFICULTY_WEIGHTS.get(label, 0.0) for label in counts)
    if code_features["for_loops"] + code_features["while_loops"] >= 2:
        difficulty_score += 0.08
    if code_features["comprehensions"]:
        difficulty_score += 0.05
    if code_features["imports"] >= 2:
        difficulty_score += 0.03

    if difficulty_score < 0.55:
        difficulty = "easy"
    elif difficulty_score < 0.80:
        difficulty = "medium"
    else:
        difficulty = "hard"

    return {
        "taxonomy": [primary] + secondary,
        "primary_taxonomy": primary,
        "secondary_taxonomy": secondary,
        "difficulty_score": round(difficulty_score, 3),
        "difficulty": difficulty,
        "signals": counts,
        "code_features": code_features,
    }

def bucket_by_difficulty(records: List[dict]) -> Dict[str, List[dict]]:
    buckets = {"easy": [], "medium": [], "hard": []}
    for rec in records:
        buckets[rec["difficulty"]].append(rec)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: (r["difficulty_score"], len(r["prompt"]), r.get("task_id") or 0))
    return buckets

def order_easy_medium_hard(records: List[dict], per_stage: int) -> List[dict]:
    buckets = bucket_by_difficulty(records)
    ordered = []
    for stage in ["easy", "medium", "hard"]:
        ordered.extend([dict(r, curriculum_stage=stage, curriculum_strategy="easy_medium_hard") for r in buckets[stage][:per_stage]])
    return ordered

def order_diversity(records: List[dict], per_stage: int) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["primary_taxonomy"]].append(rec)
    for label in grouped:
        grouped[label].sort(key=lambda r: (r["difficulty_score"], len(r["prompt"]), r.get("task_id") or 0))

    labels = sorted(grouped.keys(), key=lambda lbl: (-len(grouped[lbl]), lbl))
    selected = []
    idx = 0
    while len(selected) < per_stage and labels:
        for label in labels:
            if grouped[label]:
                rec = grouped[label].pop(0)
                selected.append(dict(rec, curriculum_stage=rec["difficulty"], curriculum_strategy="diversity"))
                if len(selected) >= per_stage:
                    break
        idx += 1
        if idx > 1000:
            break
    return selected

def order_mutate(records: List[dict], per_stage: int) -> List[dict]:
    scored = sorted(records, key=lambda r: (abs(r["difficulty_score"] - 0.55), r["difficulty_score"], len(r["prompt"])))
    chosen = []
    for rec in scored[:per_stage]:
        chosen.append(dict(rec, curriculum_stage=rec["difficulty"], curriculum_strategy="mutate"))
    return chosen

def order_all_strategies(records: List[dict], per_stage: int) -> List[dict]:
    """Ghép 3 chiến lược; mỗi bài chỉ xuất hiện MỘT lần (chiến lược chọn nó đầu tiên được giữ).

    Bản cũ khử trùng bằng chữ ký có cả tên chiến lược nên cùng một bài bị giải 2–3 lần (50 dòng chỉ 37 bài),
    chỉ khác tiền tố của curriculum_prompt.
    """
    combined: List[dict] = []
    combined.extend(order_easy_medium_hard(records, per_stage))
    combined.extend(order_diversity(records, per_stage))
    combined.extend(order_mutate(records, per_stage))

    seen = set()
    unique = []
    for rec in combined:
        signature = (str(rec.get("task_id")), compact_sentence(rec.get("prompt", "")).lower())
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(rec)
    return unique

def mutate_prompt_text(prompt: str, strategy: str, stage: str, taxonomy: Sequence[str]) -> Tuple[str, str]:
    prompt = compact_sentence(prompt)
    primary = taxonomy[0] if taxonomy else "general"

    if strategy == "easy_medium_hard":
        lead = {
            "easy": "Beginner-friendly curriculum task:",
            "medium": "Intermediate curriculum task:",
            "hard": "Advanced curriculum task:",
        }.get(stage, "Curriculum task:")
        mutated = f"{lead} {prompt}"
        notes = f"Stage-wrapped task for {stage} curriculum with primary taxonomy '{primary}'."
        return mutated, notes

    if strategy == "diversity":
        prefix = f"[Taxonomy: {primary} | Stage: {stage}]"
        mutated = f"{prefix} {prompt}"
        notes = f"Diversity-focused prompt that exposes taxonomy '{primary}' and stage '{stage}'."
        return mutated, notes

    replacements = [
        (r"\bWrite a function to\b", "Implement a Python function to"),
        (r"\bWrite a function that\b", "Implement a Python function that"),
        (r"\bGiven\b", "Consider"),
        (r"\bYou need to\b", "Your task is to"),
        (r"\bFind\b", "Determine"),
        (r"\bCompute\b", "Calculate"),
        (r"\bReturn\b", "Produce"),
    ]
    mutated = prompt
    for pattern, replacement in replacements:
        mutated = re.sub(pattern, replacement, mutated, flags=re.I)

    if mutated == prompt:
        mutated = f"Rephrased coding task: {prompt}"
    notes = f"Paraphrased the prompt while preserving semantics; taxonomy '{primary}' remains unchanged."
    return mutated, notes

def heuristic_solution_pattern(record: dict, taxonomy: Sequence[str]) -> str:
    code = record.get("reference_solution", "")
    features = infer_taxonomy(record)["code_features"]
    patterns = []
    if "dp" in taxonomy:
        patterns.append("dynamic programming / memoization")
    if "graph" in taxonomy:
        patterns.append("graph traversal")
    if "recursion" in taxonomy:
        patterns.append("recursive decomposition")
    if "dict" in taxonomy:
        patterns.append("dictionary / frequency counting")
    if "strings" in taxonomy:
        patterns.append("string normalization")
    if "lists" in taxonomy:
        patterns.append("list traversal and aggregation")
    if features["for_loops"]:
        patterns.append("iterative loops")
    if features["comprehensions"]:
        patterns.append("comprehension-based data processing")
    if not patterns:
        patterns.append("straightforward control flow")
    return ", ".join(dict.fromkeys(patterns))

def build_explanation(record: dict, curriculum_meta: dict) -> str:
    """Giải thích ngắn dựng từ taxonomy + đặc trưng của lời giải (không gọi LLM).

    Truyền record có `reference_solution` = lời giải đã kiểm chứng để mô tả đúng lời giải được lưu
    (với đề tự sinh, reference_solution ban đầu chỉ là stub).
    """
    taxonomy = curriculum_meta.get("taxonomy", ["general"])
    stage = curriculum_meta.get("curriculum_stage", "medium")
    primary = curriculum_meta.get("primary_taxonomy", "general")
    pattern = heuristic_solution_pattern(record, taxonomy)
    source = "synthetic (LLM-generated)" if str(record.get("source_file", "")).startswith("synthetic") else "dataset"
    return textwrap.dedent(
        f"""
        This {stage} {source} task focuses on {primary}.
        The stored solution uses {pattern}.
        It was accepted only after passing every assert of the task in an isolated test harness.
        """
    ).strip()

def maybe_enrich_explanation_with_llm(
    runtime: RuntimeConfig,
    record: dict,
    curriculum_meta: dict,
    curriculum_prompt: str,
    mutation_notes: str,
) -> Tuple[str, str]:
    if runtime.provider == "offline":
        return build_explanation(record, curriculum_meta), mutation_notes

    messages = [
        {
            "role": "system",
            "content": (
                "You write curriculum metadata for coding tasks. Return concise JSON only. "
                "Do not change the solution code."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_id": record.get("task_id"),
                    "strategy": curriculum_meta.get("curriculum_strategy"),
                    "stage": curriculum_meta.get("curriculum_stage"),
                    "taxonomy": curriculum_meta.get("taxonomy"),
                    "difficulty": curriculum_meta.get("difficulty"),
                    "prompt": curriculum_prompt,
                    "original_prompt": record.get("prompt"),
                    "solution_pattern": heuristic_solution_pattern(record, curriculum_meta.get("taxonomy", [])),
                    "mutation_notes": mutation_notes,
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]

    try:
        content = chat_completion(runtime, messages, temperature=0.2, max_tokens=700)
        parsed = parse_json_object(content)
        explanation = safe_strip_text(parsed.get("explanation") or parsed.get("summary") or "")
        mutated_notes = safe_strip_text(parsed.get("mutation_notes") or mutation_notes)
        if explanation:
            return explanation, mutated_notes
    except Exception:
        pass

    return build_explanation(record, curriculum_meta), mutation_notes
