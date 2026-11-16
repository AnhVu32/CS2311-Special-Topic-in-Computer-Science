import argparse
import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {line_number}: {error}") from error
    return records


def build_dataframes(records):
    full_rows = []
    prompt_answer_rows = []

    for record in records:
        task_id = record.get("task_id")
        prompt = record.get("text", "")
        answer = record.get("code", "")
        test_list = record.get("test_list", [])
        challenge_test_list = record.get("challenge_test_list", [])
        test_setup_code = record.get("test_setup_code", "")

        full_rows.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "answer": answer,
                "test_setup_code": test_setup_code,
                "test_list": json.dumps(test_list, ensure_ascii=False),
                "challenge_test_list": json.dumps(challenge_test_list, ensure_ascii=False),
            }
        )

        prompt_answer_rows.append(
            {
                "task_id": task_id,
                "prompt": prompt,
                "answer": answer,
            }
        )

    full_df = pd.DataFrame(full_rows)
    prompt_answer_df = pd.DataFrame(prompt_answer_rows)
    return full_df, prompt_answer_df


def main():
    parser = argparse.ArgumentParser(description="Convert MBPP JSONL to parquet files")
    parser.add_argument(
        "--input",
        default="mbpp.jsonl",
        help="Input MBPP JSONL file path (default: mbpp.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Output directory for parquet files (default: current directory)",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = load_jsonl(input_path)
    if not records:
        raise ValueError("Input JSONL has no valid records")

    full_df, prompt_answer_df = build_dataframes(records)

    full_path = output_dir / "mbpp_full.parquet"
    prompt_answer_path = output_dir / "mbpp_prompt_answer.parquet"

    full_df.to_parquet(full_path, index=False)
    prompt_answer_df.to_parquet(prompt_answer_path, index=False)

    print(f"Saved: {full_path} (rows={len(full_df)}, cols={list(full_df.columns)})")
    print(
        f"Saved: {prompt_answer_path} (rows={len(prompt_answer_df)}, cols={list(prompt_answer_df.columns)})"
    )


if __name__ == "__main__":
    main()
