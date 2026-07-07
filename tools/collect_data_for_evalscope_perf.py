#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遍历当前工程目录下所有深层子文件夹，提取 benchmark_percentile.json 和
benchmark_summary.json 中的关键字段，汇总到一个 Excel 文件中。

使用方法：
    python collect_benchmark.py [root_dir] [output_xlsx]

未传参时默认：
    root_dir   = 当前脚本所在目录
    output_xlsx = benchmark_summary.xlsx
"""

import json
import os
import sys
from pathlib import Path

try:
    from openpyxl import Workbook
except ImportError:
    print("缺少依赖 openpyxl，先执行: pip install openpyxl")
    sys.exit(1)


# ---------------------- 列定义 ----------------------
HEADERS = [
    "Layer1_Folder", "Layer2_Folder", "Layer3_Folder",
    "TTFT_AVG", "TTFT_P75", "TTFT_P90", "TTFT_P99",
    "TPOT_AVG", "TPOT_P75", "TPOT_P90", "TPOT_P99",
    "InputTokens_avg", "OutputTokens_avg",
    "output_tps/requst_P75", "output_tps/requst_P90", "output_tps/requst_P99",
    "Concurrency",
    "Output Token Throughput", "Total Token Throughput", "rps",
    "E2E_AVG", "E2E_P75", "E2E_P90", "E2E_P99",
    "Total Requests", "Failed Requests", "Success Requests",
]

# benchmark_summary.json 字段 -> 列名
SUMMARY_MAP = {
    "Average time to first token (s)": "TTFT_AVG",
    "Average time per output token (s)": "TPOT_AVG",
    "Average input tokens per request": "InputTokens_avg",
    "Average output tokens per request": "OutputTokens_avg",
    "Number of concurrency": "Concurrency",
    "Output token throughput (tok/s)": "Output Token Throughput",
    "Total token throughput (tok/s)": "Total Token Throughput",
    "Request throughput (req/s)": "rps",
    "Average latency (s)": "E2E_AVG",
    "Total requests": "Total Requests",
    "Failed requests": "Failed Requests",
    "Succeed requests": "Success Requests",
}

# benchmark_percentile.json: (百分位, 字段) -> 列名
PERCENTILE_MAP = {
    ("75%", "TTFT (s)"): "TTFT_P75",
    ("90%", "TTFT (s)"): "TTFT_P90",
    ("99%", "TTFT (s)"): "TTFT_P99",
    ("75%", "TPOT (s)"): "TPOT_P75",
    ("90%", "TPOT (s)"): "TPOT_P90",
    ("99%", "TPOT (s)"): "TPOT_P99",
    ("75%", "Output (tok/s)"): "output_tps/requst_P75",
    ("90%", "Output (tok/s)"): "output_tps/requst_P90",
    ("99%", "Output (tok/s)"): "output_tps/requst_P99",
    ("75%", "Latency (s)"): "E2E_P75",
    ("90%", "Latency (s)"): "E2E_P90",
    ("99%", "Latency (s)"): "E2E_P99",
}


def load_json_safely(path: Path):
    """读取 json 文件，失败时返回 None。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] 读取 {path} 失败: {e}")
        return None


def find_target_dirs(root: Path):
    """查找包含 benchmark_percentile.json 或 benchmark_summary.json 的所有目录。"""
    targets = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        if "benchmark_percentile.json" in filenames or "benchmark_summary.json" in filenames:
            targets.add(Path(dirpath))
    return sorted(targets)


def get_layer_folders(target_dir: Path, root: Path):
    """获取 target_dir 相对 root 路径的前 3 层目录名，不足处补空字符串。"""
    try:
        rel = target_dir.relative_to(root)
    except ValueError:
        rel = target_dir
    parts = list(rel.parts)
    parts += [""] * (3 - len(parts))
    return parts[:3]


def extract_summary(summary_data):
    """从 summary json 提取需要的字段。"""
    row = {}
    if not isinstance(summary_data, dict):
        return row
    for json_key, col_name in SUMMARY_MAP.items():
        if json_key in summary_data:
            row[col_name] = summary_data[json_key]
    return row


def extract_percentile(percentile_data):
    """从 percentile json 提取需要的字段。"""
    row = {}
    if not isinstance(percentile_data, list):
        return row
    # 建索引：百分位字符串 -> 该项 dict
    by_pct = {}
    for item in percentile_data:
        if isinstance(item, dict) and "Percentiles" in item:
            by_pct[str(item["Percentiles"]).strip()] = item

    for (pct, field), col_name in PERCENTILE_MAP.items():
        item = by_pct.get(pct)
        if item is not None and field in item:
            row[col_name] = item[field]
    return row


ND_MS_COLUMNS = {
    "TTFT_AVG", "TTFT_P75", "TTFT_P90", "TTFT_P99",
    "TPOT_AVG", "TPOT_P75", "TPOT_P90", "TPOT_P99",
}


def to_ms(value):
    """将秒转换为毫秒；非数值原样返回。"""
    try:
        return round(float(value) * 1000.0, 4)
    except (TypeError, ValueError):
        return value


def build_row(target_dir: Path, root: Path):
    """根据一个目标目录构建一行数据。"""
    summary_path = target_dir / "benchmark_summary.json"
    percentile_path = target_dir / "benchmark_percentile.json"

    row = {h: "" for h in HEADERS}
    layers = get_layer_folders(target_dir, root)
    row["Layer1_Folder"] = layers[0]
    row["Layer2_Folder"] = layers[1]
    row["Layer3_Folder"] = layers[2]

    if summary_path.exists():
        row.update(extract_summary(load_json_safely(summary_path)))
    if percentile_path.exists():
        row.update(extract_percentile(load_json_safely(percentile_path)))

    # 将 TTFT/TPOT 相关列由秒转换为毫秒
    for col in ND_MS_COLUMNS:
        if row.get(col, "") != "":
            row[col] = to_ms(row[col])

    return row


def main():
    if len(sys.argv) >= 2:
        root = Path(sys.argv[1]).resolve()
    else:
        root = Path(__file__).resolve().parent

    if len(sys.argv) >= 3:
        output_xlsx = Path(sys.argv[2]).resolve()
    else:
        output_xlsx = root / "summary_evalscope_perf.xlsx"

    print(f"[INFO] 扫描根目录: {root}")
    target_dirs = find_target_dirs(root)
    print(f"[INFO] 共发现 {len(target_dirs)} 个包含 benchmark 文件的目录")

    wb = Workbook()
    ws = wb.active
    ws.title = "benchmark"
    ws.append(HEADERS)

    for target_dir in target_dirs:
        row = build_row(target_dir, root)
        ws.append([row.get(h, "") for h in HEADERS])
        print(f"[OK]   {target_dir.relative_to(root)}")

    # 自适应列宽（简单版）
    for col_idx, header in enumerate(HEADERS, start=1):
        max_len = len(str(header))
        for cell in ws.iter_cols(min_col=col_idx, max_col=col_idx,
                                 min_row=2, values_only=True):
            for v in cell:
                if v is not None:
                    max_len = max(max_len, len(str(v)))
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 2, 40)

    wb.save(output_xlsx)
    print(f"[DONE] 已保存到: {output_xlsx}")


if __name__ == "__main__":
    main()