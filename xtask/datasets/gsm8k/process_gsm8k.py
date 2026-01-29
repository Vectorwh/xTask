import json
from transformers import AutoTokenizer
import argparse
import os
import logging


################################自定义函数区域#############################################
#参数化函数
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=4096, help="Number of dataset")
    parser.add_argument("--inputlen", type=int, default=2048, help="Input token lenth")
    parser.add_argument("--datasettype", type=str, default='GSM', help="dataset generator type")
    parser.add_argument("--datapath", type=str, default='/workspace/benchmark/', help="dataset sample path")
    parser.add_argument("--modelpath", type=str, default="", help="model path")
    return parser.parse_args()

#########################################################################################
#TextVQA数据集的annotation.json文件生成函数
def generate_annotations_file(batch_size, start_question_id=3602, output_file="textvqa_val_annotations.json"):
    # 创建基础结构
    data = {
        "annotations": []
    }

    # 生成指定数量的结构体
    for i in range(batch_size):
        annotation = {
          "image_id": 0,
          "answer_type": "other",
          "question_type": "other",
          "question_id": start_question_id + i,  # question_id递增
          "answers": [
            {
                  "answer_id": 1023,
                  "answer": "None",
                  "answer_confidence": "yes"
            }
          ]
        }
        data["annotations"].append(annotation)

    # 写入文件，确保格式正确
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入annotations文件head
        f.write('{\n  "annotations": [\n')

        for i, annotation in enumerate(data["annotations"]):
            # 将结构体转换为JSON字符串
            annotation_str = json.dumps(annotation, indent=2, ensure_ascii=False)
            # 调整缩进以匹配目标格式
            lines = annotation_str.split('\n')
            indented_lines = ['    ' + line for line in lines]
            indented_str = '\n'.join(indented_lines)
            # 写入结构体，最后一个不加逗号
            if i < len(data["annotations"]) - 1:
                f.write(indented_str + ',\n')
            else:
                f.write(indented_str + '\n')

        f.write('  ]\n}')

    logging.info(f"[process_dataset]: 已生成文件: {output_file}")
    logging.info(f"[process_dataset]: 包含 {batch_size} 个结构体，question_id 从 {start_question_id} 到 {start_question_id + batch_size - 1}")

#########################################################################################
#数据集问题文本生成函数
def datatmp_generator(input_len, tokenizer, dataset, num_prompts):
    dataset_tmp = []
    for sentence in dataset:
        words = tokenizer.tokenize(sentence)
        if len(words) == 0:
            continue
        len_num = len(words) // input_len
        if len_num == 0:
            multiplier = (input_len // len(words)) + 1
            repeated_len = words * multiplier
            words = repeated_len[:input_len]
            decoded_text = tokenizer.convert_tokens_to_string(words)
            dataset_tmp.append(decoded_text)
        else:
            words = words[:input_len]
            decoded_text = tokenizer.convert_tokens_to_string(words)
            dataset_tmp.append(decoded_text)

    if num_prompts < len(dataset_tmp):
        dataset_tmp = dataset_tmp[:num_prompts]
    else:
        multiplier = (num_prompts // len(dataset_tmp)) + 1
        repeat_batch = dataset_tmp * multiplier
        dataset_tmp = repeat_batch[:num_prompts]
    return dataset_tmp

################################自定义函数区域#############################################

def convert_to_int_or_default(s, back_v):
    """
    如果字符串可以转换为整数，则返回其整数值；
    否则返回默认值 1
    
    Args:
        s: 要转换的字符串
    
    Returns:
        int: 转换后的整数值或默认值 1
    """
    try:
        # 尝试转换为整数
        return int(s)
    except (ValueError, TypeError):
        # 如果转换失败（非数字字符串或类型错误），返回默认值 1
        return back_v

def process_dataset_gsm8k(batch_size, input_len, dataset_type, model_path, output_path, num_prompts):
    #调用tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    logging.info(f"[process_dataset]: 数据集生成基于该权重产生:{model_path}")
    #不同类型的数据集分类执行，函数模块反复调用;Support Dataset Type: GSM,VQA、VID
    if os.path.exists(f'{dataset_type}-in{input_len}-bs{batch_size}.jsonl'):
        logging.info("[process_dataset]: GSM8K jsonl already exists...")
        exit(0)

    dataset = []
    dataset_path = os.path.join(os.getcwd(), "xtask/datasets/gsm8k", "GSM8K.jsonl")
    with open(dataset_path, 'r', encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data['question'])

    dataset_tmp = []
    num_prompts = convert_to_int_or_default(num_prompts, batch_size)
    dataset_tmp = datatmp_generator(input_len, tokenizer, dataset, num_prompts)
    logging.info("=== 生成GSM8K.jsonl文件 ===")
    json.dumps(dataset_tmp, ensure_ascii=False, indent=4)
    output_file = os.path.join(output_path, f'GSM8K-in{input_len}-bs{batch_size}.jsonl')
    with open(output_file, 'w', encoding='utf-8') as f:
        logging.info("[process_dataset]: start generating")
        for i in range(len(dataset_tmp)):
            f.write(json.dumps({"question": dataset_tmp[i], "answer": "none"}, ensure_ascii=False))
            f.write("\n")
    logging.info(f"[process_dataset]: 已生成文件: {output_file}")

    return output_file
