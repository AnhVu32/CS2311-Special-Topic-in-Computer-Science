"""Độ trùng giữa knowledge base và MBPP+ (ô nhiễm dữ liệu). Cần: pip install evalplus scikit-learn

Chạy từ thư mục Source_Code:
    python tools/check_contamination.py ../Data/knowledge_base/all_models/knowledge_base.jsonl ../Data/knowledge_base/mbpp_seeded/knowledge_base.jsonl

- Cosine TF-IDF giữa đề MBPP+ và đề gần nhất trong KB; số bài MBPP+ vượt các ngưỡng 0.5–0.9.
- Tên hàm (entry_point) của MBPP+ xuất hiện dưới dạng `def` trong KB.
- Các bài MBPP (theo id) nằm trong KB và có thuộc MBPP+ không.
"""
import json, re, sys, collections
from evalplus.data import get_mbpp_plus
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
plus = get_mbpp_plus()
def plus_text(t):
    p = t["prompt"].strip().strip('"').strip()
    return " ".join(l for l in p.splitlines() if not l.strip().startswith("assert")).strip()
ids = list(plus); P = [plus_text(plus[i]) for i in ids]; P_fn = [plus[i]["entry_point"] for i in ids]
def load(path): return [json.loads(l) for l in open(path) if l.strip()]
def fn_names(code):
    return set(re.findall(r"^def\s+(\w+)", code or "", re.M))
for path in sys.argv[1:]:
    kb = load(path)
    K = [r.get("original_prompt") or "" for r in kb]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(P + K)
    S = cosine_similarity(vec.transform(P), vec.transform(K))
    best = S.max(axis=1); arg = S.argmax(axis=1)
    kb_fns = collections.Counter(f for r in kb for f in fn_names(r.get("solution_code")))
    same_fn = [ids[i] for i in range(len(ids)) if P_fn[i] in kb_fns]
    print(f"\n{path}: {len(kb)} rows")
    for th in (0.5, 0.6, 0.7, 0.8, 0.9):
        print(f"  MBPP+ tasks with nearest-KB cosine >= {th}: {(best >= th).sum()}")
    print(f"  MBPP+ tasks whose entry_point name appears as a def in KB: {len(same_fn)}")
    order = best.argsort()[::-1][:6]
    for i in order:
        print(f"   {best[i]:.2f} {ids[i]:9s} {P[i][:70]!r}\n        KB: {K[arg[i]][:90]!r}")
    # source ids
    src = collections.Counter((r.get("source_file"), str(r.get("source_task_id"))) for r in kb)
    mbpp_src = sorted({int(t) for (f, t) in src if f and "mbpp" in str(f) and t.isdigit()})
    if mbpp_src:
        plus_nums = {int(i.split("/")[1]) for i in ids}
        inter = [t for t in mbpp_src if t in plus_nums]
        print(f"  MBPP-derived rows: {sum(n for (f,t),n in src.items() if f and 'mbpp' in str(f))}, distinct MBPP task ids {len(mbpp_src)} (range {mbpp_src[0]}..{mbpp_src[-1]}); also in MBPP+ test set: {len(inter)} -> {inter[:15]}")
