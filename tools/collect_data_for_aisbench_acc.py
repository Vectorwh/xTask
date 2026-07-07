#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遍历当前工程文件夹下所有深层子目录, 查找形如 summary*.csv 的文件,
提取其中 dataset / mode / vllm-api-general-chat 三列的值,
并把前三层目录名作为 layer1 / layer2 / layer3, 汇总写入一个 Excel 文件。

使用方法:
    python extract_summary_to_excel.py [根目录(可选, 默认当前目录)] [输出文件(可选)]
"""

import os
import sys
import csv
import glob
from pathlib import Path

try:
    from openpyxl import Workbook
except ImportError:
    print("缺少 openpyxl, 正在尝试安装: pip install openpyxl")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openpyxl"])
    from openpyxl import Workbook


# 需要提取的列名
TARGET_COLUMNS = ["dataset", "mode", "vllm-api-general-chat"]
HEADERS = ["layer1", "layer2", "layer3"] + TARGET_COLUMNS


def find_summary_csvs(root_dir: str):
    """递归查找所有 summary*.csv 文件"""
    pattern = os.path.join(root_dir, "**", "summary*.csv")
    files = glob.glob(pattern, recursive=True)
    return sorted(files)


def extract_values_from_csv(csv_path: str):
    """从 csv 中读取目标列对应的值, 返回 (actual_chat_col, rows) 元组。
    actual_chat_col 为实际使用的列名, rows 为 dict 列表(每行一个 dict)。
    """
    rows = []
    actual_chat_col = "vllm-api-general-chat"
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                return actual_chat_col, rows
            # 字段名去前后空格匹配, 避免 BOM 或空格干扰
            fieldnames_clean = {fn.strip(): fn for fn in reader.fieldnames}

            # 确定实际使用的列名: 优先 vllm-api-general-chat, 若不存在则回退到 vllm-api-stream-chat
            FALLBACK_COLUMN = "vllm-api-stream-chat"
            PRIMARY_COLUMN = "vllm-api-general-chat"
            if PRIMARY_COLUMN in fieldnames_clean:
                actual_chat_col = PRIMARY_COLUMN
            elif FALLBACK_COLUMN in fieldnames_clean:
                actual_chat_col = FALLBACK_COLUMN
            else:
                actual_chat_col = PRIMARY_COLUMN  # 保持原逻辑, 取不到值时为空

            for row in reader:
                if not row:
                    continue
                values = {}
                for col in TARGET_COLUMNS:
                    if col == PRIMARY_COLUMN:
                        # 使用实际确定的列名来取值
                        real_key = fieldnames_clean.get(actual_chat_col, actual_chat_col)
                    else:
                        real_key = fieldnames_clean.get(col, col)
                    values[col] = row.get(real_key, "")
                # 跳过完全为空的数据行
                if any(v not in (None, "",) for v in values.values()):
                    rows.append(values)
    except Exception as e:
        print(f"[警告] 读取 {csv_path} 失败: {e}")
    return actual_chat_col, rows


def get_layer_dirs(csv_path: str, root_dir: str):
    """获取相对于 root_dir 的前 3 层目录名, 不足 3 层时用空字符串补齐"""
    rel = os.path.relpath(csv_path, root_dir)
    parts = rel.replace("\\", "/").split("/")
    # 去掉文件名, 只保留目录部分
    dir_parts = parts[:-1]
    layers = dir_parts[:3] + [""] * (3 - len(dir_parts[:3]))
    return layers


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    output_xlsx = sys.argv[2] if len(sys.argv) > 2 else "summary_aisbench_acc.xlsx"

    root_dir = os.path.abspath(root_dir)
    print(f"[INFO] 根目录: {root_dir}")
    print(f"[INFO] 输出文件: {output_xlsx}")

    csv_files = find_summary_csvs(root_dir)
    print(f"[INFO] 共找到 {len(csv_files)} 个 summary*.csv 文件")

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"
    # 表头先不写, 等确定所有列名后统一处理
    all_rows_data = []  # [(layers, actual_chat_col, records), ...]

    for csv_path in csv_files:
        layers = get_layer_dirs(csv_path, root_dir)
        actual_chat_col, records = extract_values_from_csv(csv_path)
        if not records:
            continue
        all_rows_data.append((layers, actual_chat_col, records))

    # 收集所有出现过的 chat 列名(保持出现顺序)
    chat_col_names = []
    for _, actual_chat_col, _ in all_rows_data:
        if actual_chat_col not in chat_col_names:
            chat_col_names.append(actual_chat_col)

    # 构建表头: layer1, layer2, layer3, dataset, mode, <所有chat列名>
    headers = ["layer1", "layer2", "layer3", "dataset", "mode"] + chat_col_names
    ws.append(headers)

    total_rows = 0
    for layers, actual_chat_col, records in all_rows_data:
        chat_col_offset = chat_col_names.index(actual_chat_col)
        for rec in records:
            row = layers + [rec.get("dataset", ""), rec.get("mode", "")]
            # 为所有 chat 列填空值, 仅在对应列填入实际值
            chat_values = [""] * len(chat_col_names)
            chat_values[chat_col_offset] = rec.get("vllm-api-general-chat", "")
            row += chat_values
            ws.append(row)
            total_rows += 1

    # 简单的列宽自适应
    for col_idx, header in enumerate(headers, start=1):
        max_len = len(str(header))
        for cell in ws[ws.cell(row=1, column=col_idx).column_letter]:
            v = cell.value
            if v is None:
                continue
            max_len = max(max_len, len(str(v)))
        ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 2, 60)

    wb.save(output_xlsx)
    print(f"[INFO] 完成, 共写入 {total_rows} 行数据 -> {output_xlsx}")


if __name__ == "__main__":
    main()