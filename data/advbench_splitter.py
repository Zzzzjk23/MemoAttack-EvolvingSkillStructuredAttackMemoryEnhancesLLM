from __future__ import annotations

import pandas as pd


def clean_target(value):
    if not isinstance(value, str) or not value.strip():
        return value
    text = value.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    if len(text) >= 2 and text[0] == "[" and text[-1] == "]":
        text = text[1:-1]
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        text = text[1:-1]
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return text.strip()


def main() -> None:
    csv_path = "AdvBench.csv"
    chunk_size = 50

    dataframe = pd.read_csv(csv_path)
    dataframe["target"] = dataframe["target"].astype(str).apply(clean_target)

    total_rows = len(dataframe)
    subset_count = (total_rows + chunk_size - 1) // chunk_size

    for subset_index in range(subset_count):
        start = subset_index * chunk_size
        end = min(start + chunk_size, total_rows)
        subset = dataframe.iloc[start:end]
        output_path = f"AdvBench_subset_{subset_index + 1}.csv"
        subset.to_csv(output_path, index=True)
        print(f"Wrote {output_path} with {len(subset)} rows")

    print(f"\nGenerated {subset_count} subsets from {total_rows} rows")


if __name__ == "__main__":
    main()
