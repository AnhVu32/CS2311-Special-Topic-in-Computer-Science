import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

SECRET_FLAGS = ("--api_key", "--embed_api_key")


def parse_extra_body(text: Optional[str]) -> Optional[dict]:
    """--extra_body: chuỗi JSON object -> dict (None nếu không truyền)."""
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extra_body phải là JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit('--extra_body phải là JSON object, vd \'{"reasoning_effort": "low"}\'')
    return value


def api_key_from_args(args) -> Optional[str]:
    """--api_key, hoặc đọc từ biến môi trường tên --api_key_env (không để lộ key trong dòng lệnh / meta)."""
    if getattr(args, "api_key", None):
        return args.api_key
    env_name = getattr(args, "api_key_env", None)
    if env_name:
        value = os.environ.get(env_name)
        if not value:
            raise SystemExit(f"Biến môi trường {env_name} (--api_key_env) chưa được đặt")
        return value
    return None


def sanitized_argv(argv: Optional[Sequence[str]] = None) -> List[str]:
    """sys.argv với giá trị của --api_key / --embed_api_key thay bằng *** (để ghi vào file kết quả)."""
    argv = list(sys.argv if argv is None else argv)
    out: List[str] = []
    hide = False
    for a in argv:
        if hide:
            out.append("***")
            hide = False
        elif a in SECRET_FLAGS:
            out.append(a)
            hide = True
        elif any(a.startswith(f + "=") for f in SECRET_FLAGS):
            out.append(a.split("=", 1)[0] + "=***")
        else:
            out.append(a)
    return out


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def load_jsonl_or_json(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
        return items

    if suffix == ".json":
        payload = json.loads(read_text(path))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        raise ValueError("JSON input must be a list or a {'data': [...]} object")

    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise ImportError(
                "Reading parquet requires pandas/pyarrow. Please install them first."
            ) from e
        return pd.read_parquet(path).to_dict(orient="records")

    raise ValueError("Unsupported data format. Use .jsonl, .json, or .parquet")

def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    candidates = [
        path,
        Path.cwd() / path,
    ]
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if resolved.exists():
            return resolved
    return path.resolve(strict=False)

def safe_strip_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""

def normalize_answer(answer: str) -> str:
    answer = safe_strip_text(answer).replace("$", "").replace(",", "").replace(" ", "")
    try:
        return str(int(float(answer)))
    except (ValueError, OverflowError):
        return answer

def compact_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", safe_strip_text(text)).strip()
    return text

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+.-]*)[^\n]*\n(.*?)(?:```|\Z)", flags=re.DOTALL)
_PYTHON_TAGS = {"python", "py", "python3", "py3"}


def extract_code_block(text: str) -> Optional[str]:
    """Lấy code Python từ câu trả lời markdown.

    Nhận ```python, ```py, ```python3 và fence không ghi ngôn ngữ; chấp nhận fence cuối bị cắt
    (model hết max_tokens). Ưu tiên block có 'def'/'class'; giữ quy ước cũ là lấy block cuối.
    Bỏ <think>...</think> trước khi tìm để không nhặt nhầm code trong phần suy nghĩ.
    """
    if not text:
        return None
    text = strip_think_blocks(text)
    blocks = []
    for tag, body in _FENCE_RE.findall(text):
        tag = tag.strip().lower()
        if tag and tag not in _PYTHON_TAGS:
            continue
        body = body.strip()
        if body:
            blocks.append((tag in _PYTHON_TAGS, body))
    if not blocks:
        return None
    for want_tagged in (True, False):
        with_def = [b for tagged, b in blocks if tagged == want_tagged and re.search(r"^\s*(def|class)\s", b, re.M)]
        if with_def:
            return with_def[-1]
    tagged = [b for t, b in blocks if t]
    return (tagged or [b for _, b in blocks])[-1]

def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...<truncated {len(text) - max_chars} chars>"

def parse_json_object(content: str) -> Any:
    """Parse JSON from LLM output, stripping think blocks and markdown."""
    content = strip_think_blocks(content)
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except Exception:
        # Try to find JSON array or object in the text
        for pattern in [r"\[.*\]", r"\{.*\}"]:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except Exception:
                    continue
    return None
