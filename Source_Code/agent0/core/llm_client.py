import hashlib
import json
import logging
import os
import random
import re
import socket
import threading
import time
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path
from typing import Dict, List, Optional

from agent0.core.config import DEFAULT_BASE_URLS, RuntimeConfig
import agent0.core.logger as logger_module
from agent0.core.utils import _truncate_text, safe_strip_text

logger = logging.getLogger("agent0")

# Mã HTTP coi là lỗi tạm thời -> thử lại. Các mã khác (400, 401, 404...) báo lỗi ngay.
RETRYABLE_HTTP_CODES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
DEFAULT_MAX_RETRIES = int(os.getenv("AGENT0_LLM_MAX_RETRIES", "6"))
DEFAULT_TIMEOUT = int(os.getenv("AGENT0_LLM_TIMEOUT", "300"))
MAX_BACKOFF_SECONDS = 120.0
# 429 kiểu "hết hạn mức ngày / tuần" thì không chờ được trong vài phút: báo lỗi ngay (QuotaExhaustedError),
# để lần chạy dừng sớm thay vì ngồi backoff hàng giờ; chạy lại với --resume khi hạn mức được làm mới.
QUOTA_RETRY_AFTER_SECONDS = 600.0
_QUOTA_PATTERNS = re.compile(
    r"per day|tokens per day|requests per day|\b(?:TPD|RPD)\b|daily|weekly|usage limit|insufficient_quota|"
    r"exceeded your current quota|quota exceeded|out of credits|credit balance",
    re.IGNORECASE,
)


# ───────────── đếm lời gọi + cache phản hồi (tuỳ chọn) ─────────────
# Cache: bật bằng configure_cache(dir) (cờ --cache_dir của các script). Khoá = sha256(provider, model, base_url,
# messages, temperature, max_tokens, seed). Chỉ cache lời gọi "tất định": có seed hoặc temperature <= 0.2;
# lời gọi lấy mẫu không seed (cố ý ngẫu nhiên) luôn gọi API. Chạy lại cùng lệnh -> không tốn lời gọi, ra cùng kết quả.
CACHEABLE_MAX_TEMPERATURE = 0.2
_STATS_LOCK = threading.Lock()
_CALL_STATS: Dict[str, int] = {"calls": 0, "api_calls": 0, "cache_hits": 0, "failures": 0}
_CACHE_DIR: Optional[Path] = None


def configure_cache(cache_dir: Optional[str]) -> Optional[Path]:
    """Bật (đường dẫn) / tắt (None) cache phản hồi LLM trên đĩa."""
    global _CACHE_DIR
    _CACHE_DIR = Path(cache_dir).expanduser().resolve() if cache_dir else None
    if _CACHE_DIR is not None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("LLM cache: %s", _CACHE_DIR)
    return _CACHE_DIR


def call_stats() -> Dict[str, int]:
    with _STATS_LOCK:
        return dict(_CALL_STATS)


def reset_call_stats() -> None:
    with _STATS_LOCK:
        for k in _CALL_STATS:
            _CALL_STATS[k] = 0


def _bump(key: str) -> None:
    with _STATS_LOCK:
        _CALL_STATS[key] += 1


def _cache_key(runtime: RuntimeConfig, messages: List[dict], temperature: float, max_tokens: int, seed) -> str:
    raw = json.dumps(
        {
            "provider": runtime.provider, "model": runtime.model, "base_url": runtime.base_url,
            "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed,
            "extra_body": getattr(runtime, "extra_body", None),
        },
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[str]:
    path = _CACHE_DIR / key[:2] / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["content"]
    except Exception:  # noqa: BLE001 - file cache hỏng thì coi như không có
        return None


def _cache_put(key: str, runtime: RuntimeConfig, content: str) -> None:
    path = _CACHE_DIR / key[:2] / f"{key}.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(
            {"provider": runtime.provider, "model": runtime.model, "created_at": int(time.time()), "content": content},
            ensure_ascii=False,
        ), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("Không ghi được cache LLM (%s)", exc)


class HTTPStatusError(RuntimeError):
    """Lỗi HTTP có giữ lại mã trạng thái và header Retry-After (vẫn là RuntimeError như trước)."""

    def __init__(self, code: int, url: str, body: str, retry_after: Optional[float] = None):
        super().__init__(f"HTTP {code} from {url}: {body}")
        self.code = code
        self.body = body
        self.retry_after = retry_after


class LLMCallError(RuntimeError):
    """Gọi LLM thất bại sau khi đã thử lại hết số lần cho phép."""


class QuotaExhaustedError(LLMCallError):
    """Hết hạn mức (ngày / tuần / credit) của provider — thử lại ngay không có ích."""


def is_quota_exhausted(exc: BaseException) -> bool:
    if not isinstance(exc, HTTPStatusError):
        return False
    if exc.code == 402:
        return True
    if exc.code != 429:
        return False
    if exc.retry_after is not None and exc.retry_after > QUOTA_RETRY_AFTER_SECONDS:
        return True
    return bool(_QUOTA_PATTERNS.search(exc.body or ""))


def merge_extra_body(payload: dict, extra: Optional[dict]) -> dict:
    """Gộp sâu `extra` vào payload; giá trị None xoá khoá (vd {"max_tokens": None, "max_completion_tokens": 2000})."""
    if not extra:
        return payload
    out = dict(payload)
    for key, value in extra.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_extra_body(out[key], value)
        else:
            out[key] = value
    return out


def http_get_json(url: str, timeout: int = 30, headers: Optional[dict] = None) -> dict:
    default_headers = {"User-Agent": "Agent0/1.0"}
    if headers:
        default_headers.update(headers)
    req = urllib.request.Request(url, headers=default_headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else {}


def _parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def http_post_json(url: str, payload: dict, timeout: int = 300, headers: Optional[dict] = None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "User-Agent": "Agent0/1.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        retry_after = _parse_retry_after(e.headers.get("Retry-After") if e.headers else None)
        raise HTTPStatusError(e.code, url, err[:2000], retry_after) from e


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, HTTPStatusError):
        return exc.code in RETRYABLE_HTTP_CODES and not is_quota_exhausted(exc)
    return isinstance(
        exc,
        (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, HTTPException, json.JSONDecodeError),
    )


def _backoff_seconds(attempt: int, exc: BaseException) -> float:
    """Exponential backoff có jitter; với 429 ưu tiên Retry-After của server."""
    if isinstance(exc, HTTPStatusError) and exc.retry_after is not None:
        return min(MAX_BACKOFF_SECONDS, exc.retry_after + random.uniform(0, 1))
    base = 10.0 if isinstance(exc, HTTPStatusError) and exc.code == 429 else 2.0
    delay = min(MAX_BACKOFF_SECONDS, base * (2 ** attempt))
    return delay * random.uniform(0.5, 1.0)


def post_json_with_retry(
    url: str,
    payload: dict,
    headers: Optional[dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    max_retries: Optional[int] = None,
    what: str = "request",
) -> dict:
    retries = DEFAULT_MAX_RETRIES if max_retries is None else max(0, max_retries)
    last_exc: Optional[BaseException] = None
    for attempt in range(retries + 1):
        try:
            return http_post_json(url, payload, timeout=timeout, headers=headers)
        except Exception as exc:  # noqa: BLE001 - phân loại bên dưới
            last_exc = exc
            if not _is_retryable(exc) or attempt >= retries:
                break
            wait = _backoff_seconds(attempt, exc)
            logger.warning(
                "%s lỗi tạm thời (lần %s/%s): %s — thử lại sau %.0fs",
                what, attempt + 1, retries + 1, str(exc)[:200], wait,
            )
            time.sleep(wait)
    if last_exc is not None and is_quota_exhausted(last_exc):
        raise QuotaExhaustedError(f"{what}: hết hạn mức của provider — {str(last_exc)[:500]}") from last_exc
    raise LLMCallError(f"{what} thất bại sau {attempt + 1} lần: {last_exc}") from last_exc


def _check_ollama(model: str, base_url: str) -> None:
    try:
        tags = http_get_json(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        models = [m.get("name", "") for m in tags.get("models", [])]
        if models and not any(model in m for m in models):
            print(f"WARNING: '{model}' not found in Ollama. Available: {models}")
    except Exception as e:
        raise ValueError(f"Cannot connect to Ollama at {base_url}: {e}") from e


def _base_url_for(runtime: RuntimeConfig) -> str:
    base_url = runtime.base_url or DEFAULT_BASE_URLS.get(runtime.provider)
    if not base_url:
        raise ValueError(
            f"Không có base_url cho provider '{runtime.provider}'. "
            "Tạo runtime bằng agent0.core.config.resolve_runtime(...) hoặc truyền --base_url."
        )
    return base_url.rstrip("/")


def chat_completion(
    runtime: RuntimeConfig,
    messages: List[dict],
    temperature: float = 0.2,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    max_retries: Optional[int] = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Gọi LLM (Ollama / Ollama Cloud / OpenAI-compatible).

    - Lỗi tạm thời (429, 5xx, timeout, mất kết nối) được thử lại với exponential backoff.
    - Hết số lần thử -> raise LLMCallError (không bao giờ trả về chuỗi rỗng thay cho lỗi).
    - seed được gửi kèm nếu khác None (Ollama: options.seed; OpenAI-compatible: seed).
    """
    if runtime.provider == "offline":
        raise RuntimeError("Offline mode does not support LLM chat calls")

    _bump("calls")
    cache_key = None
    if _CACHE_DIR is not None and (seed is not None or temperature <= CACHEABLE_MAX_TEMPERATURE):
        cache_key = _cache_key(runtime, messages, temperature, max_tokens, seed)
        cached = _cache_get(cache_key)
        if cached is not None:
            _bump("cache_hits")
            logger.debug("LLM cache hit | model=%s key=%s", runtime.model, cache_key[:12])
            return cached

    logger.debug(
        "LLM call start | provider=%s model=%s temperature=%s max_tokens=%s seed=%s messages=%s",
        runtime.provider, runtime.model, temperature, max_tokens, seed, len(messages),
    )
    if logger_module.LOG_LLM_IO:
        try:
            request_dump = json.dumps(messages, ensure_ascii=False, indent=2)
        except Exception:
            request_dump = str(messages)
        logger.info(
            "LLM INPUT | provider=%s model=%s\n%s",
            runtime.provider, runtime.model,
            _truncate_text(request_dump, logger_module.LLM_IO_MAX_CHARS),
        )

    base_url = _base_url_for(runtime)
    headers = {}
    if runtime.api_key and runtime.api_key != "not-used":
        headers["Authorization"] = f"Bearer {runtime.api_key}"

    if runtime.provider in ("ollama", "ollama-cloud"):
        options = {"temperature": temperature, "num_predict": max_tokens}
        if seed is not None:
            options["seed"] = int(seed)
        payload = {"model": runtime.model, "messages": messages, "stream": False, "options": options}
        payload = merge_extra_body(payload, getattr(runtime, "extra_body", None))
        _bump("api_calls")
        try:
            body = post_json_with_retry(
                f"{base_url}/api/chat", payload, headers=headers, timeout=timeout,
                max_retries=max_retries, what=f"{runtime.provider}:{runtime.model}",
            )
        except Exception:
            _bump("failures")
            raise
        content = safe_strip_text(body.get("message", {}).get("content", ""))
    else:
        # Groq / OpenAI / custom: OpenAI-compatible chat/completions
        payload = {
            "model": runtime.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = int(seed)
        payload = merge_extra_body(payload, getattr(runtime, "extra_body", None))
        _bump("api_calls")
        try:
            body = post_json_with_retry(
                f"{base_url}/chat/completions", payload, headers=headers, timeout=timeout,
                max_retries=max_retries, what=f"{runtime.provider}:{runtime.model}",
            )
        except Exception:
            _bump("failures")
            raise
        choices = body.get("choices", [])
        if not choices:
            _bump("failures")
            logger.error("LLM call failed | provider=%s no choices returned", runtime.provider)
            raise LLMCallError(f"No choices returned by {runtime.provider} API")
        content = safe_strip_text(choices[0].get("message", {}).get("content", ""))

    if logger_module.LOG_LLM_IO:
        logger.info("LLM OUTPUT | provider=%s\n%s", runtime.provider, _truncate_text(content, logger_module.LLM_IO_MAX_CHARS))
    logger.debug("LLM call done | provider=%s output_chars=%s", runtime.provider, len(content))
    if cache_key is not None and content:
        _cache_put(cache_key, runtime, content)
    return content
