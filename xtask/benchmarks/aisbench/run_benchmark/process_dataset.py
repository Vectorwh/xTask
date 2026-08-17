import json
from transformers import AutoTokenizer
import argparse
import copy
import os
import sys
import time


################################自定义函数区域#############################################
#参数化函数
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=4096, help="Number of dataset")
    parser.add_argument("--inputlen", type=int, default=2048, help="Input token lenth")
    parser.add_argument("--datasettype", type=str, default='GSM8K', help="dataset generator type")
    parser.add_argument("--datapath", type=str, default='/workspace/benchmark/', help="dataset sample path")
    parser.add_argument("--modelpath", type=str, default="", help="model path")
    parser.add_argument("--sourcepath", type=str, default="", help="source dataset path")
    parser.add_argument("--outputpath", type=str, default="", help="generated dataset path")
    parser.add_argument("--infermode", type=str, default="last", help="multi-turn inference mode")
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

    print(f"[process_dataset]: 已生成文件: {output_file}")
    print(f"[process_dataset]: 包含 {batch_size} 个结构体，question_id 从 {start_question_id} 到 {start_question_id + batch_size - 1}")

#########################################################################################
#数据集问题文本生成函数
def datatmp_generator(input_len, tokenizer, dataset, batch_size):
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

    batch_num = len(dataset_tmp) // batch_size
    if batch_num == 0:
        multiplier = (batch_size // len(dataset_tmp)) + 1
        repeat_batch = dataset_tmp * multiplier
        dataset_tmp = repeat_batch[:batch_size]
    else:
        dataset_tmp = dataset_tmp[:batch_size]
    return dataset_tmp


SHAREGPT_PADDING_UNITS = (" x", "x", " a", "a", " 0", "0", " .", ".", "\n")


def _sharegpt_messages(conversations):
    role_map = {"human": "user", "gpt": "assistant"}
    return [
        {"role": role_map[item["from"]], "content": item["value"]}
        for item in conversations[:-1]
    ]


def _sharegpt_prompt_length(tokenizer, conversations):
    token_ids = tokenizer.apply_chat_template(
        _sharegpt_messages(conversations),
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    return len(token_ids)


def _decode_token_ids(tokenizer, token_ids):
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def _repeat_token_prefix(token_ids, size):
    if size <= 0:
        return []
    multiplier = (size + len(token_ids) - 1) // len(token_ids)
    return (token_ids * multiplier)[:size]


def _find_repeated_suffix(tokenizer, conversations, prefix, unit, target_len):
    def length_for(repetitions):
        conversations[-2]["value"] = prefix + unit * repetitions
        return _sharegpt_prompt_length(tokenizer, conversations)

    current_len = length_for(0)
    if current_len == target_len:
        return prefix

    high = max(8, target_len - current_len + 8)
    max_high = max(256, target_len * 4)
    high_len = length_for(high)
    while high_len < target_len and high < max_high:
        high = min(high * 2, max_high)
        high_len = length_for(high)

    low = 0
    while low + 1 < high:
        middle = (low + high) // 2
        if length_for(middle) < target_len:
            low = middle
        else:
            high = middle

    start = max(0, low - 32)
    end = min(max_high, high + 32)
    for repetitions in range(start, end + 1):
        if length_for(repetitions) == target_len:
            return prefix + unit * repetitions
    return None


def _fit_last_human_content(tokenizer, conversations, target_len):
    original_text = conversations[-2]["value"]
    conversations[-2]["value"] = ""
    empty_prompt_len = _sharegpt_prompt_length(tokenizer, conversations)
    if empty_prompt_len > target_len:
        raise ValueError(
            f"input_len={target_len} is smaller than the model chat-template overhead "
            f"({empty_prompt_len} tokens)"
        )

    conversations[-2]["value"] = original_text
    original_prompt_len = _sharegpt_prompt_length(tokenizer, conversations)
    if original_prompt_len == target_len:
        return

    source_token_ids = tokenizer.encode(original_text, add_special_tokens=False)
    if not source_token_ids:
        source_token_ids = tokenizer.encode("ShareGPT conversation", add_special_tokens=False)
    if not source_token_ids:
        raise ValueError("the tokenizer produced no usable tokens for ShareGPT padding")

    token_budget = target_len - empty_prompt_len
    high = max(64, token_budget * 2 + 64)
    cache = {}

    def candidate_for(size):
        if size not in cache:
            text = _decode_token_ids(
                tokenizer, _repeat_token_prefix(source_token_ids, size)
            )
            conversations[-2]["value"] = text
            cache[size] = (text, _sharegpt_prompt_length(tokenizer, conversations))
        return cache[size]

    low = 0
    while low + 1 < high:
        middle = (low + high) // 2
        _, prompt_len = candidate_for(middle)
        if prompt_len <= target_len:
            low = middle
        else:
            high = middle

    if original_prompt_len < target_len:
        best_text, best_len = original_text, original_prompt_len
    else:
        best_text, best_len = "", empty_prompt_len
    candidate_sizes = {0, low, high}
    candidate_sizes.update(range(max(0, low - 64), min(high + 64, high * 2) + 1))
    for size in sorted(candidate_sizes):
        text, prompt_len = candidate_for(size)
        if prompt_len == target_len:
            conversations[-2]["value"] = text
            return
        if best_len < prompt_len < target_len:
            best_text, best_len = text, prompt_len

    conversations[-2]["value"] = best_text
    for unit in SHAREGPT_PADDING_UNITS:
        fitted_text = _find_repeated_suffix(
            tokenizer, conversations, best_text, unit, target_len
        )
        if fitted_text is not None:
            conversations[-2]["value"] = fitted_text
            return

    raise RuntimeError(
        f"unable to construct an exact {target_len}-token ShareGPT prompt; "
        f"closest prompt length was {best_len}"
    )


def _validate_sharegpt_sample(sample):
    conversations = sample.get("conversations") if isinstance(sample, dict) else None
    if not isinstance(conversations, list) or len(conversations) < 2:
        return False
    if len(conversations) % 2 != 0:
        return False
    for index, item in enumerate(conversations):
        expected_role = "human" if index % 2 == 0 else "gpt"
        if not isinstance(item, dict) or item.get("from") != expected_role:
            return False
        if not isinstance(item.get("value"), str):
            return False
    return True


def fix_sharegpt_sample(tokenizer, sample, target_len):
    fixed_sample = copy.deepcopy(sample)
    conversations = fixed_sample["conversations"]

    # infer_mode=last sends all completed history plus the final human message.
    # Drop the oldest complete rounds first so the recent context remains intact.
    while len(conversations) > 2 and _sharegpt_prompt_length(tokenizer, conversations) > target_len:
        conversations = conversations[2:]
        fixed_sample["conversations"] = conversations

    _fit_last_human_content(tokenizer, conversations, target_len)
    actual_len = _sharegpt_prompt_length(tokenizer, conversations)
    if actual_len != target_len:
        raise RuntimeError(
            f"ShareGPT prompt length validation failed: expected {target_len}, got {actual_len}"
        )
    return fixed_sample


def process_sharegpt_dataset(tokenizer, source_path, output_path, input_len, num_prompts):
    if input_len <= 0:
        raise ValueError("input_len must be a positive integer")
    if num_prompts <= 0:
        raise ValueError("num_prompts must be a positive integer")

    with open(source_path, "r", encoding="utf-8") as source_file:
        source_dataset = json.load(source_file)
    if not isinstance(source_dataset, list):
        raise ValueError("ShareGPT source dataset must be a JSON array")

    generated = []
    for sample in source_dataset:
        if len(generated) >= num_prompts:
            break
        if not _validate_sharegpt_sample(sample):
            continue
        generated.append(fix_sharegpt_sample(tokenizer, sample, input_len))

    if not generated:
        raise ValueError("ShareGPT source dataset contains no valid conversations")

    source_count = len(generated)
    while len(generated) < num_prompts:
        generated.append(copy.deepcopy(generated[len(generated) % source_count]))
    for index, sample in enumerate(generated):
        source_id = sample.get("id", "sharegpt")
        sample["id"] = f"{source_id}-xtask-{index}"

    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(generated, output_file, ensure_ascii=False)
    os.replace(temporary_path, output_path)

    for sample in generated:
        actual_len = _sharegpt_prompt_length(tokenizer, sample["conversations"])
        if actual_len != input_len:
            raise RuntimeError(
                f"generated ShareGPT validation failed: expected {input_len}, got {actual_len}"
            )
    print(
        f"[process_dataset]: generated {len(generated)} ShareGPT conversations "
        f"with exact last-prompt length {input_len}: {output_path}"
    )
    return output_path

################################自定义函数区域#############################################

if __name__ == '__main__':
    args = parse_arguments()
    # 以DeekSeek为例,权重路径更换
    batch_size = args.bs
    input_len = args.inputlen
    dataset_type = args.datasettype
    data_path = args.datapath
    model_path = args.modelpath
    #调用tokenizer
    tokenizer = AutoTokenizer.from_pretrained(f"{model_path}", trust_remote_code=True)
    print(f"[process_dataset]: 数据集生成基于该权重产生:{model_path}")
    #不同类型的数据集分类执行，函数模块反复调用;Support Dataset Type: GSM,VQA、VID
    if ( dataset_type == 'GSM8K' ):
        output_file_path = os.path.join(data_path, f'{dataset_type}-in{input_len}-bs{batch_size}.jsonl')
        if os.path.exists(output_file_path):
            print("[process_dataset]: GSM8K jsonl already exists...")
            exit(0)

        dataset = []
        # dataset_path = "./GSM8K.jsonl"
        dataset_path = os.path.join(data_path, 'GSM8K.jsonl')
        with open(dataset_path, 'r', encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                dataset.append(data['question'])

        dataset_tmp = []
        dataset_tmp = datatmp_generator(input_len, tokenizer, dataset, batch_size)
        print("=== 生成GSM8K.jsonl文件 ===")
        json_str = json.dumps(dataset_tmp, ensure_ascii=False, indent=4)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            print("[process_dataset]: start generating")
            for i in range(len(dataset_tmp)):
                f.write(json.dumps({"question": dataset_tmp[i], "answer": "none"}, ensure_ascii=False))
                f.write("\n")
        print(f"[process_dataset]: 已生成文件:{output_file_path}")

#VQA数据集生成
    elif (dataset_type == 'VQA' ):

        if os.path.exists(f'Textvqa-in{input_len}-bs{batch_size}.jsonl'):
            print("[process_dataset]: VQA dataset already exists...")
            exit(0)

        dataset = []
        vqa_question = "Explain the contents of the picture"
        dataset.append(vqa_question)

        dataset_tmp = []
        dataset_tmp = datatmp_generator(input_len, tokenizer, dataset, batch_size)
        print("=== 生成textvqa_val.json文件 ===")
        start_question_id= 34602
        json_str = json.dumps(dataset_tmp, ensure_ascii=False, indent=4)
        with open(f'Textvqa-in{input_len}-bs{batch_size}.jsonl', 'w', encoding='utf-8') as f:
            print("[process_dataset]: start generating")
            for i in range(len(dataset_tmp)):
                question_id = start_question_id + i
                f.write(json.dumps({"image": data_path, "question": dataset_tmp[i], "question_id": question_id, "answer": "None"}))
                f.write("\n")

        print(f"[process_dataset]: 已生成文件:Textvqa-in{input_len}-bs{batch_size}.jsonl")
        #生成annotations.json文件
        print("=== 生成textvqa_val_annotations.json文件中 ===")
        generate_annotations_file(batch_size, start_question_id=34602, output_file=f'Textvqa-in{input_len}-bs{batch_size}-annotation.json')

    elif dataset_type == 'SHAREGPT':
        if args.infermode.lower() != "last":
            raise ValueError("ShareGPT fixed-length preprocessing only supports infer_mode=last")
        if not args.sourcepath:
            raise ValueError("--sourcepath is required for ShareGPT preprocessing")
        output_path = args.outputpath or os.path.join(
            data_path, f"ShareGPT-in{input_len}-n{batch_size}-last.json"
        )
        process_sharegpt_dataset(
            tokenizer,
            args.sourcepath,
            output_path,
            input_len,
            batch_size,
        )


#数据集测试VID
    elif (dataset_type == 'VID' ):

        if os.path.exists(f'Videobench-in{input_len}-bs{batch_size}-qa.json'):
            print("[process_dataset]: VID dataset already exists...")
            exit(0)

        dataset = []
        video_question = "Explain the contents of the video"
        dataset.append(video_question)

        dataset_tmp = []
        dataset_tmp = datatmp_generator(input_len, tokenizer, dataset, batch_size)

        print("=== 生成videobench_qa_new.json文件中 ===")

        json_str = json.dumps(dataset_tmp, ensure_ascii=False, indent=4)
        with open(f'Videobench-in{input_len}-bs{batch_size}-qa.json', 'w', encoding='utf-8') as f:
            print("[process_dataset]: start generating")
            f.write('{\n')
            i = 0
            for i in range(len(dataset_tmp)):
                line_content = f'  "{i}": {json.dumps({"vid_path": data_path, "video_id": "1", "question": dataset_tmp[i], "choices": {"A": "win", "B": "or", "C": "lose", "D": "sss"}}, ensure_ascii=False)}'
                if i < len(dataset_tmp) - 1:
                    line_content += ','
                line_content += '\n'
                f.write(line_content)
            f.write('}')
        print(f"[process_dataset]: 已生成文件:Videobench-in{input_len}-bs{batch_size}-qa.json")

        with open(f'Videobench-in{input_len}-bs{batch_size}-answer.json', 'w', encoding='utf-8') as f:
            print("[process_dataset]: start generating")
            f.write('{\n  "TEST": {\n')
            i = 0
            for i in range(len(dataset_tmp)):
                line_content = f'    "{i}": {json.dumps({"answer": "A"})}'
                if i < len(dataset_tmp) - 1:
                    line_content += ','
                line_content += '\n'
                f.write(line_content)
            f.write('  }\n}')
        print(f"[process_dataset]: 已生成文件:Videobench-in{input_len}-bs{batch_size}-answer.json")

    #数据集测试VSD
    elif (dataset_type == 'VSD' ): 
        #使用音频数据集时需要依赖 pydub，位置调整至该if语句中。  
        from pydub import AudioSegment
        BASE_LENGTH = 15
        number = int(batch_size)
        length  = int(data_path)         
        if os.path.exists(f'./dataset/Vocalsound/{data_path}second/{batch_size}'):
            print("[process_dataset]: VSD dataset already exists...")
            exit(0)      
        if number < 2:
            raise ValueError('[peocess_dataset]: The number of the audio should be at least 2.')
        if not data_path.isdigit():
            raise ValueError('[peocess_dataset]: The length of the audio must be an integer.')
        if length < 15:
            raise ValueError('[peocess_dataset]: The length The audio length must not be less than 15 seconds.')
       
        print("=== 生成Vocalsound文件中 ===")
        audio_15s = AudioSegment.from_wav("./dataset/Vocalsound/m15_0_throatclearing.wav")
        audio_1s = audio_15s[:1000]
        audio_output = audio_15s * (length // BASE_LENGTH) + audio_1s * (length % BASE_LENGTH)
        duration_path = f'./dataset/Vocalsound/{length}second'
        concurrency_path = f'{duration_path}/{number}'
        try:
            os.mkdir(duration_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(concurrency_path)
        except FileExistsError:
            pass

        for i in range(number):
            audio_output.export(f'{concurrency_path}/m{length}_{i}_throatclearing.wav', format='wav')
        print(f"[process_dataset]: 已生成文件vocalsound测试文件，文件路径{concurrency_path}")



