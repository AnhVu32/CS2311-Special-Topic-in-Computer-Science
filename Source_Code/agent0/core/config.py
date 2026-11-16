import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent  # .../Source_Code
REPO_ROOT = SCRIPT_DIR.parent
DATA_DIR = REPO_ROOT / "Data"

# Dữ liệu gốc (chỉ đọc)
MBPP_DIR = DATA_DIR / "mbpp"
DEFAULT_DATA_FILE = MBPP_DIR / "mbpp.jsonl"
MBPP_PLUS_TASK_IDS_FILE = MBPP_DIR / "mbpp_plus_task_ids.json"

# Knowledge base do curriculum sinh ra: Data/knowledge_base/<model>/knowledge_base.jsonl
DEFAULT_OUTPUT_DIR = DATA_DIR / "knowledge_base"
DEFAULT_KB_PATH = DEFAULT_OUTPUT_DIR / "all_models" / "knowledge_base.jsonl"
DEFAULT_REJECTED_PATH = DEFAULT_OUTPUT_DIR / "rejected.jsonl"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "summary.json"

# Kết quả benchmark: Data/results/benchmark/<model>/<run>_samples.jsonl
RESULTS_DIR = DATA_DIR / "results"
DEFAULT_RESULTS_DIR = RESULTS_DIR / "benchmark"
DEFAULT_DEV_RESULTS_DIR = RESULTS_DIR / "dev"

OLLAMA_BASE_URL = "http://localhost:11434"
OPENAI_BASE_URL = "https://api.openai.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OLLAMA_CLOUD_BASE_URL = "https://ollama.com"

# llama-3.3-70b-versatile (mặc định cũ) đã bị Groq tắt cho gói Free / Developer từ 16/08/2026.
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_PROVIDER = "auto"
DEFAULT_STRATEGY = "all"

# Luật gán nhãn chủ đề (taxonomy) cho một đề, khớp trên mô tả + lời giải mẫu + test.
# Bản cũ có các mẫu quá chung (`\band\b`, `\bor\b`, `\bfor\b`, `\bwrite\b`, `\bread\b`, `\bfind\b`, `\btime\b`...):
# trên 974 bài MBPP, "io" gắn cho 100% số bài (vì "Write a function"), "loops" 59%, "bitwise" 29%.
# Bản này chỉ giữ từ khoá mang nghĩa chủ đề; vòng lặp trong code đã được tính riêng ở code_features.
TAXONOMY_RULES: Dict[str, List[str]] = {
    "math": [
        r"\bprimes?\b", r"\bfactorials?\b", r"\bgcd\b", r"\blcm\b", r"\bdivisi(?:ble|bility|ors?)\b",
        r"\bmodulo\b", r"\bfibonacci\b", r"\bpermutations?\b", r"\bcombinations?\b", r"\bbinomial\b",
        r"\bsquare root\b", r"\bperfect (?:square|number)s?\b", r"\bdigits?\b", r"\bsum of\b", r"\bproduct of\b",
        r"\bnth\b", r"\bmath\.\w+",
    ],
    "strings": [
        r"\bstrings?\b", r"\bsubstrings?\b", r"\bpalindrom\w*", r"\banagrams?\b", r"\bcharacters?\b",
        r"\bvowels?\b", r"\bwords?\b", r"\blower ?case\b", r"\bupper ?case\b", r"\.split\(", r"\.join\(",
        r"\.replace\(",
    ],
    "lists": [r"\blists?\b", r"\barrays?\b", r"\bsub-?lists?\b", r"\bsub-?arrays?\b", r"\belements?\b"],
    "tuples": [r"\btuples?\b"],
    "dict": [
        r"\bdict(?:ionary|ionaries)?\b", r"\bfrequenc(?:y|ies)\b", r"\bhash ?(?:map|table)s?\b",
        r"\boccurrences?\b", r"\bcounter\(",
    ],
    "set": [r"\bsets?\b", r"\bunique\b", r"\bduplicates?\b", r"\bintersection\b", r"\bunion\b", r"\bdistinct\b"],
    "loops": [r"\bnested loops?\b", r"\biterat(?:e|ive|ively)\b", r"\bloops?\b"],
    "recursion": [r"\brecursi(?:ve|vely|on)\b", r"\bbacktrack(?:ing)?\b", r"\btrees?\b", r"\bdepth\b"],
    "graph": [
        r"\bgraphs?\b", r"\bnodes?\b", r"\bedges?\b", r"\bbfs\b", r"\bdfs\b", r"\bshortest path\b",
        r"\bconnected components?\b", r"\bvert(?:ex|ices)\b",
    ],
    "dp": [
        r"dynamic programming", r"\bdp\b", r"\bmemoi[sz]ation\b", r"\bsubsequences?\b", r"\bknapsack\b",
        r"\btabulation\b", r"\blru_cache\b",
    ],
    "search": [r"\bsearch\b", r"\bbinary search\b", r"\bbisect\b", r"\btwo pointers?\b", r"\bsliding window\b"],
    "sorting": [
        r"\bsort(?:ed|ing|s)?\b", r"\bascending\b", r"\bdescending\b", r"\bmerge sort\b", r"\bquick ?sort\b",
        r"\bheap ?sort\b",
    ],
    "bitwise": [r"\bbitwise\b", r"\bxor\b", r"\bbits?\b", r"\bbinary representation\b", r"<<|>>"],
    "datetime": [r"\bdates?\b", r"\bdatetime\b", r"\btimestamps?\b", r"\bweekdays?\b", r"\bcalendar\b", r"\bleap years?\b"],
    "regex": [r"\bregex\b", r"\bregular expressions?\b", r"\bre\.(?:match|search|findall|finditer|sub|split|compile|fullmatch)\b"],
    "io": [r"\bfiles?\b", r"\bjson\b", r"\bcsv\b", r"\bopen\(", r"\bstdin\b", r"\binput\(\)"],
    "datastructures": [
        r"\bstacks?\b", r"\bqueues?\b", r"\bheaps?\b", r"\bheapq\b", r"\bdeque\b", r"\bpriority queues?\b",
        r"\blinked lists?\b",
    ],
}

DIFFICULTY_WEIGHTS = {
    "math": 0.10, "strings": 0.08, "lists": 0.08, "tuples": 0.06, "dict": 0.10,
    "set": 0.08, "loops": 0.05, "recursion": 0.20, "graph": 0.25,
    "dp": 0.25, "search": 0.08, "sorting": 0.08, "bitwise": 0.12,
    "datetime": 0.06, "regex": 0.16, "io": 0.05, "datastructures": 0.10,
}

PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "custom": None,
    "ollama": None,
    "offline": None,
    "ollama-cloud": "OLLAMA_API_KEY",
}

SUPPORTED_PROVIDERS = ("auto", "offline", "ollama", "ollama-cloud", "groq", "openai", "custom")

DEFAULT_BASE_URLS = {
    "ollama": OLLAMA_BASE_URL,
    "ollama-cloud": OLLAMA_CLOUD_BASE_URL,
    "groq": GROQ_BASE_URL,
    "openai": OPENAI_BASE_URL,
}

PROVIDER_BACKEND_NAMES = {
    "offline": "Offline (không gọi LLM)",
    "ollama": "Ollama local",
    "ollama-cloud": "Ollama Cloud",
    "groq": "Groq API",
    "openai": "OpenAI API",
    "custom": "OpenAI-compatible API",
}

_ENV_LOADED = False


def _parse_env_file(path: Path) -> Dict[str, str]:
    """Đọc file .env đơn giản (KEY=VALUE) khi chưa cài python-dotenv."""
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_env_files(force: bool = False) -> Optional[Path]:
    """Nạp file .env đầu tiên tìm thấy (Source_Code/.env, thư mục gốc repo, thư mục hiện tại).

    Biến môi trường đã đặt sẵn không bị ghi đè. Trả về đường dẫn file đã nạp (hoặc None).
    """
    global _ENV_LOADED
    if _ENV_LOADED and not force:
        return None
    _ENV_LOADED = True
    for candidate in [SCRIPT_DIR / ".env", REPO_ROOT / ".env", Path.cwd() / ".env"]:
        if not candidate.exists():
            continue
        if load_dotenv is not None:
            load_dotenv(candidate, override=False)
        else:
            for key, value in _parse_env_file(candidate).items():
                os.environ.setdefault(key, value)
        return candidate
    return None


@dataclass
class RuntimeConfig:
    provider: str
    model: str
    backend: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # Trường thêm vào body của mỗi request (gộp sâu; giá trị None = xoá khoá), vd
    # {"reasoning_effort": "low"} cho gpt-oss trên Groq, {"options": {"num_ctx": 8192}} cho Ollama.
    extra_body: Optional[dict] = None

# Groq tắt llama-3.1-8b-instant và llama-3.3-70b-versatile cho gói Free / Developer từ 16/08/2026
# (thay bằng openai/gpt-oss-20b, openai/gpt-oss-120b, qwen/qwen3.6-27b). Tên cũ giữ lại để nhận diện provider.
GROQ_MODELS = {
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.6-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "qwen/qwen3-235b-a22b",
}
GROQ_MODEL_PREFIXES = ("openai/", "qwen/", "meta-llama/", "moonshotai/")


def detect_provider(model: str, base_url: Optional[str] = None) -> str:
    """Chọn provider khi người dùng để 'auto' (logic lấy từ bản CS2230, có bổ sung Ollama Cloud).

    - Có --base_url            -> custom (API tương thích OpenAI)
    - Model của Groq           -> groq nếu có GROQ_API_KEY
    - Tên dạng 'gemma3:12b'    -> ollama-cloud nếu có OLLAMA_API_KEY, ngược lại Ollama local
    - Còn lại                  -> groq / openai nếu có key tương ứng, cuối cùng là Ollama local
    """
    if base_url:
        return "custom"
    if (model in GROQ_MODELS or model.startswith(GROQ_MODEL_PREFIXES)) and os.getenv("GROQ_API_KEY"):
        return "groq"
    if ":" in model:
        return "ollama-cloud" if os.getenv("OLLAMA_API_KEY") else "ollama"
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def resolve_runtime(
    provider: Optional[str],
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> RuntimeConfig:
    """Tạo RuntimeConfig đầy đủ: nạp .env, chọn provider, điền base_url và api_key.

    Thứ tự ưu tiên: tham số dòng lệnh > biến môi trường / .env > giá trị mặc định của provider.
    Báo lỗi rõ ràng khi thiếu API key thay vì để lỗi xảy ra ở lần gọi LLM đầu tiên.
    """
    load_env_files()
    provider = (provider or DEFAULT_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Provider không hợp lệ: {provider!r}. Chọn một trong {', '.join(SUPPORTED_PROVIDERS)}")
    if provider == "auto":
        provider = detect_provider(model, base_url)

    if provider == "offline":
        return RuntimeConfig(provider="offline", model=model, backend=PROVIDER_BACKEND_NAMES["offline"])

    if provider == "custom" and not base_url:
        raise ValueError("Provider 'custom' cần --base_url (endpoint tương thích OpenAI, vd http://localhost:8000/v1)")

    resolved_base_url = base_url or DEFAULT_BASE_URLS.get(provider)
    env_key = PROVIDER_ENV_KEYS.get(provider)
    resolved_api_key = api_key or (os.getenv(env_key) if env_key else None)
    if env_key and not resolved_api_key:
        raise ValueError(
            f"Thiếu {env_key} cho provider '{provider}'. Đặt {env_key}=... trong file .env "
            f"(xem .env.example ở thư mục gốc) hoặc truyền --api_key."
        )
    if provider == "custom" and not resolved_api_key:
        resolved_api_key = "not-used"

    return RuntimeConfig(
        provider=provider,
        model=model,
        backend=PROVIDER_BACKEND_NAMES.get(provider, provider),
        base_url=resolved_base_url,
        api_key=resolved_api_key,
    )


def describe_runtime(runtime: RuntimeConfig) -> Dict[str, Optional[str]]:
    """Thông tin runtime để ghi log / summary (không bao giờ ghi api_key)."""
    return {
        "provider": runtime.provider,
        "model": runtime.model,
        "backend": runtime.backend,
        "base_url": runtime.base_url,
        "api_key_set": bool(runtime.api_key and runtime.api_key != "not-used"),
    }
