"""Tách một KB đã gắn p_hat thành hai KB cùng cỡ để kiểm tra cơ chế "đề ở biên năng lực" của Agent0.

KB đầu vào phải được sinh với `run_curriculum --uncertainty_samples k --p_hat_action tag` (giữ mọi đề đã kiểm chứng,
chỉ ghi p_hat). Chạy từ thư mục Source_Code:

    python tools/split_kb_by_phat.py /kaggle/working/kb_new/qwen2.5-coder_7b/knowledge_base.jsonl \\
        --out_dir /kaggle/working/kb_split

Ghi ra:
  <out_dir>/phat/knowledge_base.jsonl   các mục có p_hat trong --band (mặc định [0.25, 0.75] như Agent0)
  <out_dir>/ctrl/knowledge_base.jsonl   cùng số mục, chọn ngẫu nhiên (seed cố định) trong mọi mục có p_hat
                                        (--control out_of_band: chỉ chọn trong các mục ngoài khoảng)
  <out_dir>/split_report.json           số mục, phân bố p_hat của từng KB, số mục chung

Benchmark hai KB bằng cùng một cấu hình (vd `run_matrix --configs retrieved --kb_tag phat` rồi `--kb_tag ctrl`):
chênh lệch giữa hai lần chạy là tác dụng của việc chọn đề theo p_hat, vì nguồn đề, model sinh và cỡ KB như nhau.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def read_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_repaired(row: dict) -> bool:
    return bool(row.get("tests_modified")) or bool(row.get("test_repair_rounds_used"))


def p_hat_histogram(rows) -> dict:
    return dict(sorted(Counter(str(r.get("p_hat")) for r in rows).items()))


def split(rows, band=(0.25, 0.75), control="all", seed=0, drop_test_repaired=True):
    """Trả về (in_band, control_rows, thống kê)."""
    low, high = band
    stats = {"rows": len(rows)}
    tagged = [r for r in rows if r.get("p_hat") is not None]
    stats["without_p_hat"] = len(rows) - len(tagged)
    if drop_test_repaired:
        before = len(tagged)
        tagged = [r for r in tagged if not test_repaired(r)]
        stats["dropped_test_repaired"] = before - len(tagged)
    in_band = [r for r in tagged if low <= float(r["p_hat"]) <= high]
    pool = tagged if control == "all" else [r for r in tagged if not low <= float(r["p_hat"]) <= high]
    size = min(len(in_band), len(pool))
    rng = random.Random(f"split_kb_by_phat:{seed}")
    ctrl = rng.sample(pool, size) if size else []
    if size < len(in_band):  # ít mục đối chứng hơn: cắt KB phat cho bằng cỡ
        in_band = rng.sample(in_band, size)
    in_uids = {r.get("uid") for r in in_band}
    stats.update({
        "tagged": len(tagged), "in_band": len(in_band), "control": len(ctrl), "control_pool": len(pool),
        "overlap": sum(1 for r in ctrl if r.get("uid") in in_uids),
        "p_hat_all": p_hat_histogram(tagged), "p_hat_phat": p_hat_histogram(in_band), "p_hat_ctrl": p_hat_histogram(ctrl),
    })
    return in_band, ctrl, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kb", help="knowledge_base.jsonl đã gắn p_hat (--p_hat_action tag)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--band", type=float, nargs=2, default=[0.25, 0.75], metavar=("LOW", "HIGH"))
    ap.add_argument("--control", choices=("all", "out_of_band"), default="all",
                    help="all: đối chứng chọn ngẫu nhiên trong mọi mục (= không lọc theo p_hat); "
                         "out_of_band: chỉ trong các mục quá dễ / quá khó")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep_test_repaired", action="store_true", help="Giữ mục có test từng bị sửa cho khớp lời giải")
    ap.add_argument("--min_size", type=int, default=20, help="Báo lỗi nếu mỗi KB ít hơn chừng này mục")
    args = ap.parse_args()

    rows = read_rows(Path(args.kb))
    in_band, ctrl, stats = split(rows, tuple(args.band), args.control, args.seed, not args.keep_test_repaired)
    if stats["tagged"] == 0:
        print("KB không có mục nào có p_hat: sinh lại với --uncertainty_samples k --p_hat_action tag", file=sys.stderr)
        return 2
    out = Path(args.out_dir)
    write_rows(out / "phat" / "knowledge_base.jsonl", in_band)
    write_rows(out / "ctrl" / "knowledge_base.jsonl", ctrl)
    report = {"source": str(args.kb), "band": args.band, "control": args.control, "seed": args.seed, **stats}
    (out / "split_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "tagged", "in_band", "control", "overlap")}, ensure_ascii=False))
    if len(in_band) < args.min_size:
        print(f"Cảnh báo: mỗi KB chỉ có {len(in_band)} mục (< {args.min_size}) — kết quả so sánh sẽ rất nhiễu; "
              "sinh thêm đề (tăng --synthetic_count).", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
