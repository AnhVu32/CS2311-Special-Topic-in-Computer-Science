"""Độ giống văn bản giữa các đề (thuần Python, không cần scikit-learn).

Dùng cho:
- khoá dedup: `prompt_key()` — chuẩn hoá chữ thường + khoảng trắng;
- khử ô nhiễm: đề mới quá giống một đề MBPP+ (cosine TF-IDF >= ngưỡng) thì không đưa vào KB;
- khử gần-trùng: đề mới gần như trùng một đề đã có trong KB.

TF-IDF: unigram + bigram, sublinear tf, idf làm mượt (như scikit-learn), bỏ stopword và các chữ chung của đề
lập trình ("write", "function", "python", "find"...), bỏ đuôi số nhiều. Điểm KHÔNG trùng khớp với
tools/check_contamination.py (scikit-learn, stopword khác). Hiệu chỉnh trên MBPP (596 bài ngoài MBPP+ so với 378 bài MBPP+):
  >= 0.7   gần như chép lại (MBPP có bài trùng nguyên văn, vd 922 ≡ Mbpp/415, 924 ≡ Mbpp/309): 47 bài;
  0.6–0.7  bài "anh em" chỉ khác một chi tiết (chẵn/lẻ, dương/âm, vị trí đầu/cuối, cube/cuboid): +32 bài;
  0.5–0.6  cùng bài nhưng diễn đạt khác (KB all_models: "number of divisors" ~ Mbpp/96 0.60, "longest palindromic
           subsequence" ~ Mbpp/247 0.55, "contiguous subarray with the largest sum" ~ Mbpp/559 0.51) lẫn bài anh em.
Ngưỡng khử ô nhiễm mặc định 0.5: bỏ nhầm một đề tự sinh gần như không tốn gì, còn để lọt một bài MBPP+ có lời
giải đúng vào KB thì kết quả benchmark mất giá trị. Diễn đạt lại dài hơn nữa vẫn có thể lọt (~0.49) — kiểm tra
thêm bằng tools/check_contamination.py.
"""
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9_]+")

# Bộ stopword nhỏ: đủ để các chữ như "write a function to" không làm đề nào cũng giống nhau.
STOPWORDS = frozenset(
    """a an the and or of to in on at for from by with without into onto over under than then that this these those
    is are was were be been being it its as if else not no yes do does did done can could should would will shall may
    might must have has had which who whom whose what when where why how all any each every some such only own same
    so too very just also both either neither nor but while about above below between after before again further once
    here there you your we our they their he she his her them i me my given given write function python program code
    return returns returning find finds""".split()
)


def prompt_key(text: Optional[str]) -> str:
    """Khoá so trùng: chữ thường, gộp khoảng trắng, bỏ dấu câu ở hai đầu."""
    return _WS.sub(" ", (text or "").strip().lower()).strip(" .:;!?\"'`")


def _stem(token: str) -> str:
    """Bỏ đuôi số nhiều / ngôi thứ ba đơn giản (numbers -> number, calculates -> calculate, boxes -> box)."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("es") and token[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: Optional[str]) -> List[str]:
    return [_stem(t) for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS]


def _ngrams(tokens: Sequence[str], ngram_range: Tuple[int, int]) -> Counter:
    out: Counter = Counter()
    lo, hi = ngram_range
    for n in range(lo, hi + 1):
        for i in range(len(tokens) - n + 1):
            out[" ".join(tokens[i : i + n])] += 1
    return out


class TfidfIndex:
    """Chỉ mục TF-IDF nhỏ trong bộ nhớ. idf học trên `fit_docs` (mặc định chính các tài liệu được đánh chỉ mục)."""

    def __init__(
        self,
        docs: Iterable[str] = (),
        ids: Optional[Iterable] = None,
        fit_docs: Optional[Iterable[str]] = None,
        ngram_range: Tuple[int, int] = (1, 2),
    ):
        self.ngram_range = ngram_range
        docs = list(docs)
        ids = list(ids) if ids is not None else list(range(len(docs)))
        if len(ids) != len(docs):
            raise ValueError("ids và docs phải cùng độ dài")
        fit = list(fit_docs) if fit_docs is not None else docs
        df: Counter = Counter()
        for d in fit:
            df.update(set(_ngrams(tokenize(d), ngram_range)))
        self.n_fit = len(fit)
        # idf làm mượt như scikit-learn: log((1 + N) / (1 + df)) + 1
        self.idf: Dict[str, float] = {t: math.log((1 + self.n_fit) / (1 + c)) + 1.0 for t, c in df.items()}
        self.unknown_idf = math.log(1 + self.n_fit) + 1.0
        self.ids: List = []
        self.texts: List[str] = []
        self.vectors: List[Dict[str, float]] = []
        for i, d in zip(ids, docs):
            self.add(d, i)

    def vector(self, text: str) -> Dict[str, float]:
        tf = _ngrams(tokenize(text), self.ngram_range)
        vec = {t: (1.0 + math.log(c)) * self.idf.get(t, self.unknown_idf) for t, c in tf.items()}
        norm = math.sqrt(sum(w * w for w in vec.values()))
        return {t: w / norm for t, w in vec.items()} if norm else {}

    def add(self, text: str, doc_id=None) -> None:
        self.ids.append(doc_id if doc_id is not None else len(self.ids))
        self.texts.append(text)
        self.vectors.append(self.vector(text))

    def __len__(self) -> int:
        return len(self.vectors)

    @staticmethod
    def _dot(a: Dict[str, float], b: Dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b.get(t, 0.0) for t, w in a.items())

    def most_similar(self, text: str) -> Tuple[float, Optional[object], Optional[str]]:
        """(cosine lớn nhất, id, văn bản) — (0.0, None, None) nếu chỉ mục rỗng."""
        q = self.vector(text)
        best, best_i = 0.0, -1
        for i, v in enumerate(self.vectors):
            s = self._dot(q, v)
            if s > best:
                best, best_i = s, i
        if best_i < 0:
            return 0.0, None, None
        return best, self.ids[best_i], self.texts[best_i]


def mbpp_plus_reference(data_file, plus_ids: Iterable[int]) -> Tuple[List[str], List[str], List[str]]:
    """Đề MBPP+ lấy từ file MBPP gốc (cùng mô tả với MBPP+ v0.2.0), không cần cài evalplus.

    Trả về (id dạng 'Mbpp/<n>', mô tả các bài MBPP+, mô tả mọi bài MBPP để học idf).
    """
    plus = set(int(i) for i in plus_ids)
    ids, texts, all_texts = [], [], []
    with Path(data_file).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = (row.get("text") or row.get("prompt") or "").strip()
            all_texts.append(text)
            try:
                tid = int(row.get("task_id"))
            except (TypeError, ValueError):
                continue
            if tid in plus:
                ids.append(f"Mbpp/{tid}")
                texts.append(text)
    return ids, texts, all_texts


def build_mbpp_plus_index(data_file, plus_ids: Iterable[int]) -> TfidfIndex:
    ids, texts, all_texts = mbpp_plus_reference(data_file, plus_ids)
    return TfidfIndex(texts, ids=ids, fit_docs=all_texts + texts)
