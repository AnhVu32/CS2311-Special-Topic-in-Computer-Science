# Đổi MBPP sang parquet

Script `convert_mbpp_to_parquet.py` đọc `mbpp.jsonl` và ghi hai file parquet:

- `mbpp_full.parquet`: đủ các trường của MBPP, test lưu dạng chuỗi JSON.
- `mbpp_prompt_answer.parquet`: schema rút gọn cho pipeline huấn luyện / đánh giá.

Chạy từ thư mục `Source_Code` (cần `pip install pandas pyarrow`):

```bash
python tools/convert_mbpp_to_parquet.py --input ../Data/mbpp/mbpp.jsonl --output_dir ../Data/mbpp
```
