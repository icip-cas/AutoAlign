import pandas as pd
import uuid
import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Convert parquet file to JSON with UUID.")
    parser.add_argument(
        "--parquet_file",
        type=str,
        required=True,
        help="Path to the input parquet file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="output.json",
        help="Path to the output JSON file (default: output.json)"
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.parquet_file)

    result = [
        {
            "prompt": row["prompt"],
            "uuid": str(uuid.uuid4())
        }
        for _, row in df.iterrows()
    ]

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()