import json


def write_list_to_jsonl(data_list: list, output_file: str):
    """
    将列表数据逐行写入JSONL文件
    
    Args:
        data_list: 包含字典的列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data_list:
            # 转换为JSON字符串并写入，确保添加换行符
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"✅ 已写入 {len(data_list)} 行到 {output_file}")


def jsonl_to_json(jsonl_file, json_file):
    """将 JSONL 文件转换为 JSON 数组"""
    data = []

    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                data_line = json.loads(line)
                json_data_line = {"inputs_pretokenized": data_line.get("prompt")}
                data.append(json_data_line)

    # 写入 JSON 文件
    write_list_to_jsonl(data, json_file)

source_jsonl = 'b_search_query_v4_sampled_500.jsonl'
target_jsonl = 'b_search_query_v4_sampled_500_1.jsonl'

# 使用示例
jsonl_to_json(source_jsonl, target_jsonl)