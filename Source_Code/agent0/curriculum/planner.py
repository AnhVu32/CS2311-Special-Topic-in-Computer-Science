import json
import logging
import math
from typing import Any, Dict, List, Optional

from agent0.core.config import RuntimeConfig
from agent0.core.utils import parse_json_object
from agent0.core.llm_client import chat_completion

logger = logging.getLogger("agent0")

SYSTEM_MSG = {
    "role": "system",
    "content": (
        "You are a curriculum designer for coding education. "
        "Return strict JSON only, no markdown. No explanation."
    ),
}

def _parse_subtopics(content: str) -> List[dict]:
    parsed = parse_json_object(content)
    if isinstance(parsed, dict):
        items = (parsed.get("subtopics") or parsed.get("new_subtopics")
                 or parsed.get("plan") or parsed.get("data") or [])
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    valid = []
    for st in items:
        if isinstance(st, dict) and st.get("name"):
            st.setdefault("task_count", 1)
            st.setdefault("difficulty", "medium")
            st.setdefault("description", st["name"])
            valid.append(st)
    return valid

def _as_count(value) -> int:
    try:
        return max(1, int(float(value)))
    except (TypeError, ValueError):
        return 1


def _adjust_task_counts(subtopics: List[dict], target: int) -> List[dict]:
    """Chia đúng `target` đề theo tỉ lệ task_count (phương pháp phần dư lớn nhất).

    Bản cũ làm tròn từng subtopic lên tối thiểu 1 rồi dồn phần chênh vào subtopic đầu -> có thể ra số âm
    (vd 20 subtopic x 2, target 30 -> subtopic đầu = -8). Giờ tổng luôn bằng target, không có số âm;
    subtopic được 0 đề thì bỏ.
    """
    if target <= 0 or not subtopics:
        return []
    weights = [_as_count(s.get("task_count", 1)) for s in subtopics]
    total = sum(weights)
    raw = [w * target / total for w in weights]
    counts = [int(math.floor(r)) for r in raw]
    order = sorted(range(len(raw)), key=lambda i: (raw[i] - counts[i], weights[i]), reverse=True)
    for i in order[: target - sum(counts)]:
        counts[i] += 1
    return [dict(s, task_count=c) for s, c in zip(subtopics, counts) if c > 0]

def _feedback_instructions(feedback: Optional[List[dict]]) -> List[str]:
    """Phản hồi từ vòng trước (tỉ lệ giải đúng p_hat theo subtopic) — thay cho bước cập nhật RL của Agent0."""
    if not feedback:
        return []
    lines = [
        "Feedback from the previous round: p_hat is the fraction of sampled solver attempts that passed the tests "
        "(the goal is tasks the solver solves about half of the time, p_hat around 0.5).",
    ]
    for f in feedback[:40]:
        lines.append(
            f"- {f.get('subtopic')} ({f.get('difficulty', '?')}): p_hat={f.get('p_hat_mean')} over {f.get('n')} tasks"
        )
    lines.append("For subtopics with p_hat > 0.8 plan harder variants; for p_hat < 0.2 plan easier or clearer ones; "
                 "keep subtopics near 0.5 and add new ones of similar difficulty.")
    return lines


def _initial_plan(runtime: RuntimeConfig, domain: str, count: int, feedback: Optional[List[dict]] = None) -> List[dict]:
    prompt = {
        "role": "user",
        "content": json.dumps(
            {
                "task": "Plan subtopics for a coding knowledge base",
                "domain": domain,
                "total_tasks": count,
                "instructions": [
                    f"Create a diverse curriculum plan with subtopics covering the domain '{domain}'.",
                    "For each subtopic, specify: name, description, difficulty (easy/medium/hard), and task_count.",
                    f"The sum of all task_count must equal {count}.",
                    "Cover a wide range: basics, data structures, algorithms, string manipulation, math, etc.",
                    "Return a JSON array of objects with keys: name, description, difficulty, task_count.",
                ] + _feedback_instructions(feedback),
            },
            ensure_ascii=False,
            indent=2,
        ),
    }

    logger.info("Step 1 — Initial plan | domain='%s' total_tasks=%s", domain, count)
    content = chat_completion(runtime, [SYSTEM_MSG, prompt])
    subtopics = _parse_subtopics(content)

    if not subtopics:
        raise ValueError("LLM failed to create initial plan — returned empty result")

    logger.info("Step 1 done | %s subtopics", len(subtopics))
    return subtopics

def _reflect_on_plan(runtime: RuntimeConfig, domain: str, count: int, current: List[dict], round_num: int) -> List[dict]:
    current_names = [s["name"] for s in current]

    prompt = {
        "role": "user",
        "content": json.dumps(
            {
                "task": "Identify MISSING topics and add them to the curriculum",
                "domain": domain,
                "existing_subtopics": current_names,
                "instructions": [
                    f"The current plan has these subtopics: {current_names}.",
                    "Your job: identify important topics that are MISSING from this list.",
                    "Return ONLY the NEW subtopics as a JSON array (do NOT repeat existing ones).",
                    "Each new subtopic: name, description, difficulty (easy/medium/hard), task_count (1-3).",
                    "Add 3-6 new subtopics that are genuinely missing.",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
    }

    logger.info("Step 2 — Reflection round %s | reviewing %s subtopics", round_num, len(current))
    content = chat_completion(runtime, [SYSTEM_MSG, prompt])
    new_subtopics = _parse_subtopics(content)

    existing_lower = {n.lower() for n in current_names}
    added = [s for s in new_subtopics if s["name"].lower() not in existing_lower]

    if added:
        logger.info("Step 2 — Reflection %s done | +%s new subtopics", round_num, len(added))
        return current + added

    logger.warning("Step 2 — Reflection %s found no new subtopics to add", round_num)
    return current

def plan_curriculum(
    runtime: RuntimeConfig,
    domain: str = "coding",
    total_tasks: int = 30,
    reflection_rounds: int = 1,
    feedback: Optional[List[dict]] = None,
) -> List[dict]:
    subtopics = _initial_plan(runtime, domain, total_tasks, feedback)

    for r in range(1, reflection_rounds + 1):
        try:
            subtopics = _reflect_on_plan(runtime, domain, total_tasks, subtopics, r)
        except Exception as e:
            # lỗi tạm thời (429, 5xx) đã được llm_client thử lại với backoff
            logger.warning("Reflection round %s failed: %s", r, e)

    subtopics = _adjust_task_counts(subtopics, total_tasks)

    logger.info(
        "Planning complete | %s subtopics | plan=%s",
        len(subtopics),
        [(s["name"], s["task_count"], s["difficulty"]) for s in subtopics],
    )
    return subtopics
