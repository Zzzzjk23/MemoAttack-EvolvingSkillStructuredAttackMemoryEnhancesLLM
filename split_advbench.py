"""
读取 AdvBench.csv，按顺序每 50 行生成一个 AdvBench_subset_n.csv，
并清理 target 列：去除最外层双引号、方括号、单引号。
"""
import pandas as pd
import re


def clean_target(s):
    """去除 target 最外层的双引号、方括号、单引号。"""
    if not isinstance(s, str) or not s.strip():
        return s
    t = s.strip()
    # 最外层双引号
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1]
    # 最外层方括号
    if len(t) >= 2 and t[0] == '[' and t[-1] == ']':
        t = t[1:-1]
    # 最外层单引号
    if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
        t = t[1:-1]
    # 若清理后仍是双引号包裹（如 ["..."] 被读成 "...）
    if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
        t = t[1:-1]
    return t.strip()


def main():
    csv_path = "AdvBench.csv"
    chunk_size = 50

    df = pd.read_csv(csv_path)
    # 清理 target 列
    df["target"] = df["target"].astype(str).apply(clean_target)

    n_rows = len(df)
    n_subsets = (n_rows + chunk_size - 1) // chunk_size

    for i in range(n_subsets):
        start = i * chunk_size
        end = min(start + chunk_size, n_rows)
        subset = df.iloc[start:end]
        out_path = f"AdvBench_subset_{i + 1}.csv"
        subset.to_csv(out_path, index=True)
        print(f"已写入 {out_path}，行数: {len(subset)}")

    print(f"\n共生成 {n_subsets} 个子集，总行数 {n_rows}")


if __name__ == "__main__":
    main()
