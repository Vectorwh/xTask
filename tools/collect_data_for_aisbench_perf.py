import csv
import os
import glob
import re
import json  # 用于解析JSON文件


def get_excel_col_index(col_letter):
    """
    将Excel列字母转换为0开始的索引
    例如: A=0, B=1, C=2, D=3
    """
    return ord(col_letter.upper()) - ord('A')


def detect_delimiter(line):
    """
    自动检测行的分隔符。
    优先检查 '|'，否则检查 ','，最后检查 '\t'。
    """
    if '|' in line:
        parts = line.split('|')
        if len(parts) > 1: return '|', parts

    if ',' in line:
        parts = line.split(',')
        if len(parts) > 1: return ',', parts

    if '\t' in line:
        parts = line.split('\t')
        if len(parts) > 1: return '\t', parts

    return None, [line]


def parse_and_extract(file_path, root_dir):
    """
    读取单个文件，自动识别格式并提取指定指标数值
    同时提取同名的json文件中的数据
    """
    filename = os.path.basename(file_path)

    # 1. 解析文件夹层级
    rel_path = os.path.relpath(file_path, root_dir)
    path_parts = rel_path.split(os.sep)

    # 默认值为 "N/A"
    layer1_name = "N/A"
    layer2_name = "N/A"
    layer3_name = "N/A"

    if len(path_parts) >= 2:
        layer1_name = path_parts[0]
    if len(path_parts) >= 3:
        layer2_name = path_parts[1]
    if len(path_parts) >= 4:
        layer3_name = path_parts[2]

    # 2. 定义CSV提取映射关系
    # 注意：此处键名即为输出CSV的列名
    extract_mapping = {
        # TTFT (Row 3)
        'TTFT_AVG': 'C3',
        'TTFT_MINI': 'D3',
        'TTFT_MAX': 'E3',
        'TTFT_P75': 'G3',
        'TTFT_P90': 'H3',
        'TTFT_P99': 'I3',
        # TPOT (Row 4)
        'TPOT_AVG': 'C4',
        'TPOT_MINI': 'D4',
        'TPOT_MAX': 'E4',
        'TPOT_P75': 'G4',
        'TPOT_P90': 'H4',
        'TPOT_P99': 'I4',
        # Tokens (Row 6, 7, 8)
        'InputTokens_avg': 'C6',
        'OutputTokens_avg': 'C7',
        'output_tps_avg': 'C8',
        # E2E (Row 2) - 新增
        'E2E_AVG': 'C2',
        'E2E_Min': 'D2',
        'E2E_MAX': 'E2',
        'E2E_P90': 'H2'
    }

    # 初始化结果字典
    extracted_data = {
        'Layer1_Folder': layer1_name,
        'Layer2_Folder': layer2_name,
        'Layer3_Folder': layer3_name,
        'Filename': filename
    }

    # 初始化CSV列默认值
    for header in extract_mapping.keys():
        extracted_data[header] = "N/A"

    # 3. 定义JSON提取映射关系
    # 格式: "JSON中的键名": "输出CSV中的列名"
    json_mapping = {
        "Concurrency": "Concurrency",
        "Max Concurrency": "Max Concurrency",  # 新增
        "Input Token Throughput": "Input Token Throughput",
        "Output Token Throughput": "Output Token Throughput",
        "Total Token Throughput": "Total Token Throughput",
        "Request Throughput": "rps",
        "Total Requests": "Total Requests",  # 新增
        "Failed Requests": "Failed Requests",  # 新增
        "Success Requests": "Success Requests"  # 新增
    }

    # 初始化JSON列默认值
    for csv_col in json_mapping.values():
        extracted_data[csv_col] = "N/A"

    # ------------------- CSV 解析部分 -------------------
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.strip().splitlines()
        data_matrix = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line: continue

            if i == 0:
                line = re.sub(r'^{.*?：', '', line).strip()

            if i == len(lines) - 1:
                line = line.rstrip('}').strip()

            delimiter, row_raw = detect_delimiter(line)
            row = [col.strip() for col in row_raw]

            if row:
                data_matrix.append(row)

        # 智能列偏移检测
        # 如果源文件包含行号列(如 "0 | Performance...")，会导致实际的Excel列C对应数组索引3(偏移+1)
        # 我们通过检查第一行是否有数字索引来判断是否需要偏移
        col_offset = 0
        if data_matrix:
            first_cell = data_matrix[0][0]
            # 如果第一个单元格是纯数字，说明有索引列，Excel列号与数组索引有偏移
            if first_cell.isdigit():
                col_offset = 1

        # 提取CSV数据
        for header, cell in extract_mapping.items():
            col_str = ''.join(filter(str.isalpha, cell))
            row_str = ''.join(filter(str.isdigit, cell))

            if not col_str or not row_str: continue

            try:
                # 计算基础列索引
                base_col_idx = get_excel_col_index(col_str)
                # 应用偏移量：Excel C列(索引2) -> 若有偏移则指向数组索引3
                actual_col_idx = base_col_idx + col_offset

                row_idx = int(row_str) - 1

                # 检查索引范围并提取
                if 0 <= row_idx < len(data_matrix):
                    row_data = data_matrix[row_idx]
                    if 0 <= actual_col_idx < len(row_data):
                        extracted_data[header] = row_data[actual_col_idx]
                    else:
                        extracted_data[header] = "ColError"
                else:
                    extracted_data[header] = "RowError"

            except Exception:
                extracted_data[header] = "ParseError"

    except Exception as e:
        print(f"Error reading CSV {filename}: {e}")

    # ------------------- JSON 解析部分 -------------------
    try:
        # 构建json文件路径：与csv同目录
        json_file_path = os.path.join(os.path.dirname(file_path), 'gsm8k.json')

        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as jf:
                json_data = json.load(jf)

            # 根据映射关系提取字段
            for json_key, csv_col in json_mapping.items():
                if json_key in json_data:
                    val = json_data[json_key].get("total", "N/A")
                    extracted_data[csv_col] = val

    except Exception as e:
        print(f"Error reading JSON for {filename}: {e}")

    return extracted_data


def main():
    # 配置路径
    input_dir = './'
    output_file = 'summary_aisbench_perf.csv'

    # 获取文件列表：递归查找所有子目录下的gsm8k.csv
    search_pattern = os.path.join(input_dir, '**', 'gsm8k.csv')
    all_files = glob.glob(search_pattern, recursive=True)

    all_files = [f for f in all_files if not f.endswith(output_file)]

    if not all_files:
        print(f"在路径 '{input_dir}' 及其子目录下未找到任何 gsm8k.csv 文件。")
        return

    print(f"找到 {len(all_files)} 个文件，开始处理...")

    results = []
    for file_path in all_files:
        data = parse_and_extract(file_path, input_dir)
        results.append(data)
        print(
            f"已处理: Layer1=[{data['Layer1_Folder']}], Layer2=[{data['Layer2_Folder']}], Layer3=[{data['Layer3_Folder']}]")

    # 保存结果
    # 表头顺序：层级文件夹 + 文件名 + CSV指标 + JSON指标 (按指定顺序)
    fieldnames = [
        'Layer1_Folder', 'Layer2_Folder', 'Layer3_Folder', 'Filename',
        # TTFT & TPOT
        'TTFT_AVG', 'TTFT_MINI', 'TTFT_MAX', 'TTFT_P75', 'TTFT_P90', 'TTFT_P99',
        'TPOT_AVG', 'TPOT_MINI', 'TPOT_MAX', 'TPOT_P75', 'TPOT_P90', 'TPOT_P99',
        # Tokens
        'InputTokens_avg', 'OutputTokens_avg', 'output_tps_avg',
        # JSON (Concurrency组)
        'Concurrency', 'Max Concurrency',
        'Input Token Throughput', 'Output Token Throughput', 'Total Token Throughput',
        'rps',
        # E2E (新增在rps之后)
        'E2E_AVG', 'E2E_Min', 'E2E_MAX', 'E2E_P90',
        # 请求统计 (新增在最后)
        'Total Requests', 'Failed Requests', 'Success Requests'
    ]

    try:
        with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n提取完成！结果已保存至: {output_file}")
    except IOError as e:
        print(f"写入结果文件失败: {e}")


if __name__ == "__main__":
    main()