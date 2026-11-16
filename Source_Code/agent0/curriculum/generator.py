import json
import logging
import random
import re
from typing import Any, List, Optional

from agent0.core.config import RuntimeConfig
from agent0.core.llm_client import chat_completion
from agent0.core.utils import parse_json_object
from agent0.data.mbpp_loader import normalize_llm_synthetic_task
from agent0.curriculum.planner import plan_curriculum

logger = logging.getLogger("agent0")

def _generate_tasks_for_subtopic(
    runtime: RuntimeConfig,
    subtopic: dict,
    start_idx: int,
    seed: int,
) -> List[dict]:
    task_count = subtopic["task_count"]
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert Python programming instructor. Return strictly a JSON array of task objects. "
                "No markdown formatting. Do NOT provide the solution code yet."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instruction": "Generate Python programming exercises.",
                    "topic": subtopic["name"],
                    "description": subtopic["description"],
                    "difficulty": subtopic["difficulty"],
                    "count": task_count,
                    "seed": seed,
                    "requirements": [
                        "Return a JSON array where each object has: 'prompt', 'function_name', 'test_list', and 'challenge_test_list'.",
                        "The 'prompt' should clearly state the problem and inputs/outputs.",
                        "Provide 3-5 assert tests in test_list.",
                        "Optionally provide 0-2 harder asserts in challenge_test_list.",
                        "Every element of test_list and challenge_test_list must be ONE Python assert statement that calls the function, e.g. \"assert add(2, 3) == 5\".",
                        "Each task must be a self-contained, deterministic function: no network access, no file or console I/O, no threads or processes, no randomness or current time, no user input.",
                        "No markdown wrappers; JSON array only.",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]

    # seed được gửi tới API (bản cũ chỉ ghi seed vào nội dung prompt nên không tái lập được)
    content = chat_completion(runtime, messages, temperature=0.5, max_tokens=3500, seed=seed)

    # Strip <think>...</think> blocks
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    parsed: Any
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = parse_json_object(content)
        if parsed is None:
            parsed = []

    if isinstance(parsed, dict):
        items = parsed.get("tasks") or parsed.get("data") or []
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    collected: List[dict] = []
    idx = start_idx
    for raw in items:
        normalized = normalize_llm_synthetic_task(raw, idx)
        if normalized:
            normalized["subtopic"] = subtopic["name"]
            normalized["subtopic_difficulty"] = subtopic["difficulty"]
            collected.append(normalized)
            idx += 1
            if len(collected) >= task_count:
                break

    return collected

def generate_synthetic_records_with_llm(
    runtime: RuntimeConfig,
    count: int,
    seed: int = 42,
    domain: str = "coding",
    feedback: Optional[List[dict]] = None,
) -> List[dict]:
    if runtime.provider == "offline":
        raise ValueError("LLM synthetic generation requires non-offline provider")

    logger.info(
        "Synthetic LLM generation start | domain=%s count=%s provider=%s model=%s",
        domain,
        count,
        runtime.provider,
        runtime.model,
    )

    # Phase 1: LLM plans subtopics
    subtopics = plan_curriculum(runtime, domain, count, feedback=feedback)

    # Phase 2: LLM generates tasks per subtopic
    collected: List[dict] = []
    global_idx = 1

    for i, subtopic in enumerate(subtopics, start=1):
        logger.info(
            "Phase 2: Generating tasks | subtopic=%s (%s/%s) difficulty=%s task_count=%s",
            subtopic["name"],
            i,
            len(subtopics),
            subtopic["difficulty"],
            subtopic["task_count"],
        )

        for attempt in range(3):
            try:
                tasks = _generate_tasks_for_subtopic(
                    runtime, subtopic, global_idx, seed + i + attempt
                )
                break
            except Exception as e:
                # Lỗi tạm thời (429, 5xx, timeout) đã được llm_client thử lại với backoff; ở đây chỉ thử lại
                # với seed khác khi model trả JSON hỏng.
                logger.warning(
                    "Phase 2 failed | subtopic=%s attempt=%s error=%s",
                    subtopic["name"],
                    attempt + 1,
                    e,
                )
                if attempt == 2:
                    tasks = []

        logger.info(
            "Phase 2 done | subtopic=%s generated=%s/%s",
            subtopic["name"],
            len(tasks),
            subtopic["task_count"],
        )
        collected.extend(tasks)
        global_idx += len(tasks)

    if len(collected) < count:
        logger.warning(
            "Synthetic LLM generation partial | collected=%s/%s",
            len(collected),
            count,
        )
        if len(collected) == 0:
            raise ValueError("Could not generate any synthetic tasks via LLM.")
    else:
        collected = collected[:count]

    random.seed(seed)
    random.shuffle(collected)
    logger.info("Synthetic LLM generation completed | final_count=%s", len(collected[:count]))
    return collected[:count]
