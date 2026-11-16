"""Đóng gói project thành một file zip để tải lên Kaggle làm Dataset (không kèm .env / API key).

Chạy từ bất kỳ đâu:
    python3 "Source_Code/tools/package_for_kaggle.py"                      # -> <thư mục cha của project>/CS2311_kaggle_upload.zip
    python3 Source_Code/tools/package_for_kaggle.py --out ~/Desktop/x.zip --include_results

Có trong zip: README.md, requirements.txt, .env.example, Source_Code/ (kể cả kaggle/), Data/mbpp/, Data/knowledge_base/.
Bỏ: .env, __pycache__, .DS_Store, cache embedding (*.index.json), Report/, Slide/ (PDF không cần trên Kaggle),
Data/results/ (trừ khi --include_results). Notebook tìm project trong /kaggle/input dù Kaggle giải nén zip hay không.
"""
import argparse
import fnmatch
import sys
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
EXCLUDE_NAMES = {".env", ".DS_Store", "__pycache__", ".git", ".gradio", ".ipynb_checkpoints"}
EXCLUDE_GLOBS = ("*.pyc", "*.index.json", "*_padded.jsonl", "*_padded_eval_results.json", "*.tmp")


def wanted(rel: Path, include_results: bool) -> bool:
    parts = rel.parts
    if any(p in EXCLUDE_NAMES for p in parts):
        return False
    if any(fnmatch.fnmatch(rel.name, g) for g in EXCLUDE_GLOBS):
        return False
    top = parts[0]
    if top in ("README.md", "requirements.txt", ".env.example", "Source_Code"):
        return True
    if top == "Data":
        if len(parts) > 1 and parts[1] == "results":
            return include_results
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="Mặc định: <thư mục cha của project>/CS2311_kaggle_upload.zip")
    ap.add_argument("--include_results", action="store_true", help="Kèm Data/results (kết quả cũ, chỉ để đối chiếu)")
    args = ap.parse_args()

    out = Path(args.out).expanduser() if args.out else PROJECT.parent / "CS2311_kaggle_upload.zip"
    out = out.resolve()
    if out.is_relative_to(PROJECT):
        raise SystemExit("--out phải nằm ngoài thư mục project")
    root_name = PROJECT.name
    files = sorted(p for p in PROJECT.rglob("*") if p.is_file() and wanted(p.relative_to(PROJECT), args.include_results))
    if not any(p.name == "__init__.py" and p.parent.name == "agent0" for p in files):
        raise SystemExit(f"Không thấy Source_Code/agent0 trong {PROJECT}")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".zip.tmp")
    total = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            zf.write(p, f"{root_name}/{p.relative_to(PROJECT).as_posix()}")
            total += p.stat().st_size
    tmp.replace(out)
    print(f"{len(files)} file ({total / 1e6:.1f} MB trước nén) -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("Tải file này lên Kaggle: Datasets -> New Dataset (để Private). Xem Source_Code/kaggle/README.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
