# Data

```
Data/
├── mbpp/                       dữ liệu gốc, chỉ đọc
│   ├── mbpp.jsonl              974 bài MBPP (đủ các split)
│   ├── sanitized-mbpp.json     bản đã kiểm tay của MBPP
│   ├── test.jsonl              split test của MBPP
│   ├── mbpp_plus_task_ids.json 378 task_id + entry_point của MBPP+ v0.2.0 (dùng để kiểm tra file samples, tách tập dev)
│   └── README.md               mô tả bộ MBPP
├── knowledge_base/             KB do run_curriculum sinh ra
│   ├── <model>/                knowledge_base.jsonl, rejected.jsonl, summary.json, summary_history.jsonl
│   │                           (+ knowledge_base.index.json là cache embedding, khoá theo hash nội dung)
│   ├── all_models/             KB gộp các model (485 mục) — KB mặc định của benchmark và demo
│   ├── all_models_clean/       all_models kiểm lại bằng harness mới (tools/clean_kb.py): 431 mục
│   │                           + knowledge_base.report.json (lý do loại từng nhóm)
│   └── mbpp_seeded/            KB sinh từ đề MBPP gốc (bản cũ, chưa lọc): 210 dòng lấy từ MBPP, trong đó 18 bài
│                               thuộc MBPP+ (12 bài trùng nguyên văn) -> KHÔNG dùng để benchmark MBPP+.
│                               Lần chạy mới với --source mbpp ghi vào mbpp_seeded/<model>/ và đã bỏ các bài MBPP+.
└── results/
    ├── benchmark/<model>/      kết quả MBPP+ hợp lệ (đủ 378 bài, gần như không có stub)
    ├── benchmark/_invalid/     các lần chạy hỏng, giữ để đối chiếu (xem README trong đó)
    ├── dev/<model>/            kết quả chọn n trên tập dev (lệnh run_benchmark dev)
    └── rescored_linux/         bảng chấm lại toàn bộ bằng EvalPlus trên Linux
```

Đường dẫn mặc định nằm trong `Source_Code/agent0/core/config.py`.
