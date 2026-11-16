"""Truy hồi ví dụ few-shot từ knowledge base (KB) bằng embedding (Ollama `/api/embed`).

- Mỗi mục KB được nhúng theo đề gốc (`original_prompt`). Embedding lưu ở `<kb>.index.json`, khoá theo hash nội
  dung (sha1 của embed_model + đề): sửa KB mà số dòng không đổi vẫn nhúng lại đúng các mục bị sửa, thêm mục mới
  chỉ nhúng phần mới. (Bản cũ chỉ so `entry_count` nên dùng nhầm embedding cũ.)
- Khử trùng theo đề đã chuẩn hoá (bản cũ: 150 ký tự đầu).
- Bộ lọc khi nạp: bỏ mục sinh từ chính các bài đang được đánh giá (`exclude_source_task_ids`, `exclude_prompts`),
  tuỳ chọn bỏ mục mà test đã bị sửa (`exclude_test_repaired`).
- `query()`: ngưỡng độ giống `min_similarity` (không mục nào đạt -> 0 ví dụ, tức zero-shot), MMR tuỳ chọn để các ví
  dụ không na ná nhau.
- `format_few_shot()`: 'tests' (đề + assert ví dụ + lời giải, cùng định dạng prompt MBPP+), 'full' (đề + reasoning
  + lời giải, như bản cũ), 'code' (đề + lời giải).
"""
import hashlib
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from agent0.core.llm_client import http_post_json
from agent0.data.text_similarity import prompt_key

logger = logging.getLogger("agent0")

FEWSHOT_FORMATS = ("tests", "full", "code")
INDEX_VERSION = 2
EMBED_BATCH = 32


def get_embedding(
    text: str,
    model: str = "mxbai-embed-large",
    base_url: str = "http://localhost:11434",
    api_key: Optional[str] = None,
) -> List[float]:
    return get_embeddings_batch([text], model=model, base_url=base_url, api_key=api_key)[0]


def get_embeddings_batch(
    texts: List[str],
    model: str = "mxbai-embed-large",
    base_url: str = "http://localhost:11434",
    api_key: Optional[str] = None,
) -> List[List[float]]:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "input": texts}
    result = http_post_json(f"{base_url.rstrip('/')}/api/embed", payload, headers=headers, timeout=120)

    embeddings = result.get("embeddings", [])
    if len(embeddings) != len(texts) or any(not e for e in embeddings):
        raise RuntimeError(f"Expected {len(texts)} embeddings from {model}, got {len(embeddings)}")
    return embeddings


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine(a: Sequence[float], b: Sequence[float], na: Optional[float] = None, nb: Optional[float] = None) -> float:
    na = _norm(a) if na is None else na
    nb = _norm(b) if nb is None else nb
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


# giữ tên cũ cho mã bên ngoài
_cosine_similarity = _cosine


def _mbpp_number(task_id) -> Optional[int]:
    try:
        return int(str(task_id).split("/")[-1])
    except (TypeError, ValueError):
        return None


def is_mbpp_derived(row: dict) -> bool:
    """Mục KB sinh từ một bài MBPP (seed là đề MBPP), không phải đề tự sinh."""
    src = str(row.get("source_file") or "").lower()
    return "mbpp" in src and _mbpp_number(row.get("source_task_id")) is not None


def assert_example_lines(tests: Sequence[str], n_asserts: int = 1, max_lines: int = 6) -> List[str]:
    """Các dòng test đầu tiên cho tới khi có đủ n_asserts dòng assert (giữ dòng lệnh setup đứng trước)."""
    out: List[str] = []
    seen = 0
    for line in tests or []:
        line = (line or "").strip()
        if not line:
            continue
        out.append(line)
        if line.startswith("assert"):
            seen += 1
            if seen >= n_asserts:
                break
        if len(out) >= max_lines:
            break
    return out if seen else []


class KnowledgeRetriever:
    def __init__(
        self,
        kb_path: str,
        embed_model: str = "mxbai-embed-large",
        embed_base_url: str = "http://localhost:11434",
        embed_api_key: Optional[str] = None,
        exclude_source_task_ids: Optional[Iterable] = None,
        exclude_prompts: Optional[Iterable[str]] = None,
        exclude_test_repaired: bool = False,
        row_filter: Optional[Callable[[dict], bool]] = None,
    ):
        self.kb_path = Path(kb_path)
        self.index_path = self.kb_path.with_suffix(".index.json")
        self.embed_model = embed_model
        self.embed_base_url = embed_base_url
        self.embed_api_key = embed_api_key
        self.exclude_source_ids = {n for n in (_mbpp_number(t) for t in (exclude_source_task_ids or [])) if n is not None}
        self.exclude_prompt_keys = {prompt_key(p) for p in (exclude_prompts or []) if p}
        self.exclude_test_repaired = exclude_test_repaired
        self.row_filter = row_filter

        self.entries: List[dict] = []
        self.embeddings: List[List[float]] = []
        self._norms: List[float] = []
        self._hashes: List[str] = []
        self._keys: set = set()
        self._loaded = False
        self._embedded = False
        self.stats: Dict[str, int] = {}

    # ───────── nạp KB ─────────

    @staticmethod
    def _entry_from_row(row: dict) -> dict:
        return {
            "original_prompt": row.get("original_prompt") or row.get("curriculum_prompt") or "",
            "solution_code": row.get("solution_code", ""),
            "reasoning": row.get("reasoning", ""),
            "difficulty": row.get("difficulty", "unknown"),
            "taxonomy": row.get("taxonomy", []),
            "uid": row.get("uid", ""),
            "test_list": row.get("test_list", []) or [],
            "test_setup_code": row.get("test_setup_code", "") or "",
            "source_task_id": row.get("source_task_id"),
            "source_model": row.get("source_model"),
            "p_hat": row.get("p_hat"),
        }

    @staticmethod
    def _test_repaired(row: dict) -> bool:
        return bool(row.get("tests_modified")) or bool(row.get("test_repair_rounds_used"))

    def _load_kb(self) -> List[dict]:
        stats = {"rows": 0, "kept": 0, "invalid": 0, "duplicates": 0, "excluded_source": 0, "excluded_prompt": 0,
                 "excluded_test_repaired": 0, "excluded_filter": 0}
        self.stats = stats
        if not self.kb_path.exists():
            logger.warning("KB not found: %s", self.kb_path)
            return []
        entries: List[dict] = []
        seen = set()
        with self.kb_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats["rows"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid"] += 1
                    continue
                entry = self._entry_from_row(row)
                if not entry["original_prompt"] or not entry["solution_code"] or row.get("accepted") is False:
                    stats["invalid"] += 1
                    continue
                key = prompt_key(entry["original_prompt"])
                if self.exclude_source_ids and is_mbpp_derived(row) and _mbpp_number(row.get("source_task_id")) in self.exclude_source_ids:
                    stats["excluded_source"] += 1
                    continue
                if key in self.exclude_prompt_keys:
                    stats["excluded_prompt"] += 1
                    continue
                if self.exclude_test_repaired and self._test_repaired(row):
                    stats["excluded_test_repaired"] += 1
                    continue
                if self.row_filter is not None and not self.row_filter(row):
                    stats["excluded_filter"] += 1
                    continue
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
                entries.append(entry)
        stats["kept"] = len(entries)
        self._keys = seen
        logger.info("Loaded KB | %s", ", ".join(f"{k}={v}" for k, v in stats.items() if v))
        return entries

    # ───────── index embedding ─────────

    def _content_hash(self, text: str) -> str:
        return hashlib.sha1(f"{self.embed_model}\n{text}".encode("utf-8")).hexdigest()

    def _load_index_cache(self) -> Dict[str, List[float]]:
        if not self.index_path.exists():
            return {}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - index chỉ là cache
            logger.warning("Index hỏng, dựng lại (%s)", exc)
            return {}
        if data.get("version") != INDEX_VERSION or data.get("embed_model") != self.embed_model:
            logger.info("Index cũ hoặc khác model embedding -> dựng lại")
            return {}
        return dict(zip(data.get("hashes", []), data.get("embeddings", [])))

    def _save_index(self) -> None:
        data = {
            "version": INDEX_VERSION,
            "embed_model": self.embed_model,
            "kb_path": str(self.kb_path),
            "entry_count": len(self.entries),
            "hashes": self._hashes,
            "embeddings": self.embeddings,
        }
        tmp = self.index_path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, self.index_path)
        logger.info("Saved index | %s embeddings to %s", len(self.embeddings), self.index_path)

    def load_entries(self) -> int:
        """Chỉ nạp KB (không cần server embedding) — đủ cho chế độ chọn ví dụ ngẫu nhiên / cố định."""
        self.entries = self._load_kb()
        self._loaded = True
        return len(self.entries)

    def build_index(self, force: bool = False) -> int:
        self.load_entries()
        if not self.entries:
            self.embeddings, self._norms, self._hashes = [], [], []
            self._embedded = True
            return 0
        cache = {} if force else self._load_index_cache()
        hashes = [self._content_hash(e["original_prompt"]) for e in self.entries]
        missing = [i for i, h in enumerate(hashes) if h not in cache]
        self.stats["embedded_cached"] = len(hashes) - len(missing)
        self.stats["embedded_new"] = len(missing)
        if missing:
            logger.info("Embedding %s/%s mục KB (model=%s)", len(missing), len(hashes), self.embed_model)
        for start in range(0, len(missing), EMBED_BATCH):
            batch = missing[start:start + EMBED_BATCH]
            vectors = get_embeddings_batch(
                [self.entries[i]["original_prompt"] for i in batch],
                model=self.embed_model, base_url=self.embed_base_url, api_key=self.embed_api_key,
            )
            for i, v in zip(batch, vectors):
                cache[hashes[i]] = v
        self._hashes = hashes
        self.embeddings = [cache[h] for h in hashes]
        self._norms = [_norm(v) for v in self.embeddings]
        if missing or not self.index_path.exists():
            self._save_index()
        self._embedded = True
        logger.info("Index ready | %s entries (%s mới nhúng)", len(self.entries), len(missing))
        return len(self.entries)

    def add_entry(self, entry: dict, save: bool = True) -> bool:
        """Thêm một mục vào index trong bộ nhớ (không ghi vào KB). save=False để ghi index một lần ở cuối."""
        if not self._embedded:
            self.build_index()
        row = self._entry_from_row(entry)
        if not row["original_prompt"] or not row["solution_code"]:
            return False
        key = prompt_key(row["original_prompt"])
        if key in self._keys:
            return False
        emb = get_embedding(row["original_prompt"], model=self.embed_model, base_url=self.embed_base_url,
                            api_key=self.embed_api_key)
        self._keys.add(key)
        self.entries.append(row)
        self.embeddings.append(emb)
        self._norms.append(_norm(emb))
        self._hashes.append(self._content_hash(row["original_prompt"]))
        if save:
            self._save_index()
        return True

    # ───────── truy vấn ─────────

    def rewrite_query(self, prompt: str, runtime_config: Optional[dict] = None) -> str:
        if not runtime_config:
            return prompt

        from agent0.core.config import RuntimeConfig
        from agent0.core.llm_client import chat_completion

        rc = RuntimeConfig(
            provider=runtime_config.get("provider", "ollama"),
            model=runtime_config.get("model", "gemma3:4b"),
            backend=runtime_config.get("backend", "ollama"),
            base_url=runtime_config.get("base_url", "http://localhost:11434"),
            api_key=runtime_config.get("api_key"),
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the coding task below into a clearer, more searchable form. "
                    "Add related algorithm/technique keywords. "
                    "Return only the rewritten query, one paragraph, no code."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            rewritten = chat_completion(rc, messages, temperature=0.1, max_tokens=200)
            if rewritten:
                logger.info("Query rewritten | original=%s... -> rewritten=%s...", prompt[:60], rewritten[:60])
                return rewritten
        except Exception as e:  # noqa: BLE001
            logger.warning("Query rewrite failed: %s", e)
        return prompt

    def _result(self, idx: int, sim: Optional[float]) -> dict:
        e = self.entries[idx]
        out = dict(e)
        out["similarity"] = round(sim, 4) if sim is not None else None
        return out

    def query(
        self,
        prompt: str,
        n: int = 3,
        runtime_config: Optional[dict] = None,
        min_similarity: float = 0.0,
        mmr_lambda: float = 1.0,
    ) -> List[dict]:
        """n mục giống nhất (cosine >= min_similarity). mmr_lambda < 1: chọn theo MMR
        (lambda * giống-đề - (1 - lambda) * giống-ví-dụ-đã-chọn)."""
        if not self._embedded:
            self.build_index()
        if not self.entries or not self.embeddings or n <= 0:
            return []

        search_prompt = self.rewrite_query(prompt, runtime_config) if runtime_config else prompt
        q = get_embedding(search_prompt, model=self.embed_model, base_url=self.embed_base_url, api_key=self.embed_api_key)
        nq = _norm(q)
        scored = sorted(
            ((_cosine(q, emb, nq, ne), idx) for idx, (emb, ne) in enumerate(zip(self.embeddings, self._norms))),
            reverse=True,
        )
        candidates = [(s, i) for s, i in scored if s >= min_similarity]
        if mmr_lambda >= 1.0 or len(candidates) <= 1:
            picks = candidates[:n]
        else:
            pool = candidates[: max(20, 4 * n)]
            picks: List[Tuple[float, int]] = []
            while pool and len(picks) < n:
                best = None
                for s, i in pool:
                    redundancy = max(
                        (_cosine(self.embeddings[i], self.embeddings[j], self._norms[i], self._norms[j]) for _, j in picks),
                        default=0.0,
                    )
                    score = mmr_lambda * s - (1.0 - mmr_lambda) * redundancy
                    if best is None or score > best[0]:
                        best = (score, s, i)
                picks.append((best[1], best[2]))
                pool = [(s, i) for s, i in pool if i != best[2]]
        return [self._result(i, s) for s, i in picks]

    def sample(self, n: int, rng: random.Random) -> List[dict]:
        """n mục ngẫu nhiên (không cần embedding) — cho ablation 'ví dụ ngẫu nhiên' / 'ví dụ cố định'."""
        if not self._loaded:
            self.load_entries()
        if not self.entries or n <= 0:
            return []
        idxs = rng.sample(range(len(self.entries)), min(n, len(self.entries)))
        return [self._result(i, None) for i in idxs]

    # ───────── định dạng ─────────

    def format_few_shot(self, examples: List[dict], style: str = "full", n_asserts: int = 1) -> str:
        if not examples:
            return ""
        if style not in FEWSHOT_FORMATS:
            raise ValueError(f"few-shot format không hợp lệ: {style} (chọn {FEWSHOT_FORMATS})")
        parts = []
        for i, ex in enumerate(examples, 1):
            prompt = (ex.get("original_prompt") or "").strip()
            code = (ex.get("solution_code") or "").strip()
            if style == "tests":
                lines = assert_example_lines(ex.get("test_list") or [], n_asserts=n_asserts)
                problem = "\n".join([prompt] + lines)
                parts.append(f'### Example {i}\n```\n"""\n{problem}\n"""\n```\nSolution:\n```python\n{code}\n```')
            elif style == "code":
                parts.append(f"### Example {i}\nTask: {prompt}\nSolution:\n```python\n{code}\n```")
            else:
                reasoning_block = f"Reasoning:\n{ex['reasoning']}\n" if ex.get("reasoning") else ""
                parts.append(f"### Example {i}\nTask: {prompt}\n{reasoning_block}Solution:\n```python\n{code}\n```")
        return "\n\n".join(parts)
