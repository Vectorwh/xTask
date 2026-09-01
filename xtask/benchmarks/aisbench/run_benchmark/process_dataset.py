import json
from transformers import AutoTokenizer
import argparse
import copy
import heapq
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
    parser.add_argument("--prefix-len", type=int, default=512,
                        help="minimum shared prompt prefix length in tokens")
    parser.add_argument("--prefix-groups", type=int, default=4,
                        help="number of independent prefix groups")
    parser.add_argument("--prefix-source", type=str, default="dataset",
                        choices=("dataset",),
                        help="source of the shared prefix")
    parser.add_argument("--prefix-assembly", type=str, default="single",
                        choices=("single", "packed"),
                        help="build the prefix from one or multiple conversations")
    parser.add_argument("--prefix-seed-requests", type=int, choices=(0, 1), default=1,
                        help="put one seed request per group at the beginning")
    parser.add_argument("--prefix-order", type=str, choices=("grouped", "interleaved"),
                        default="grouped", help="request ordering between groups")
    parser.add_argument("--prefix-min-units", type=int, default=None,
                        help="minimum source-unit pool size for packed assembly (defaults to prefix_groups)")
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
    return len(_sharegpt_prompt_token_ids(tokenizer, conversations))


def _sharegpt_prompt_token_ids(tokenizer, conversations):
    # Prefer the tokenizer's own chat_template when available so we stay
    # consistent with inference-time tokenization. Fall back to an explicit
    # simple template for tokenizers that ship none (e.g. DeepSeek-V4),
    # which otherwise raise "no chat_template was specified" on apply.
    simple_template = (
        "{% for message in messages %}"
        "{{ message['role'] | capitalize }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "Assistant:"
    )
    messages = _sharegpt_messages(conversations)
    try:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    except (ValueError, TypeError):
        token_ids = tokenizer.apply_chat_template(
            messages,
            chat_template=simple_template,
            tokenize=True,
            add_generation_prompt=True,
        )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    # transformers >=5 may return a BatchEncoding dict (keys: input_ids /
    # attention_mask) instead of a plain token-id list when an explicit
    # chat_template is passed with tokenize=True. Unwrap so downstream
    # common-prefix math sees real token ids, not dict keys.
    if isinstance(token_ids, dict) or hasattr(token_ids, "keys"):
        token_ids = token_ids["input_ids"]
    return list(token_ids)


def _common_prefix_length(token_id_lists):
    if not token_id_lists:
        return 0
    common_length = min(len(token_ids) for token_ids in token_id_lists)
    for index in range(common_length):
        token = token_id_lists[0][index]
        if any(token_ids[index] != token for token_ids in token_id_lists[1:]):
            return index
    return common_length


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


def _sharegpt_pair_sources(samples):
    """Return unique user/assistant pairs that may become final turns."""
    pair_sources = []
    seen_questions = set()
    for sample_index, sample in enumerate(samples):
        conversations = sample["conversations"]
        for turn_index in range(0, len(conversations), 2):
            question = conversations[turn_index]["value"]
            if question in seen_questions:
                continue
            seen_questions.add(question)
            pair_sources.append(
                {
                    "source_index": sample_index,
                    "source_id": sample.get("id", f"sharegpt-{sample_index}"),
                    "turn_index": turn_index // 2,
                    "question": question,
                    "answer": conversations[turn_index + 1]["value"],
                }
            )
    return pair_sources


def _iter_sharegpt_prefix_candidates(tokenizer, samples, prefix_len, input_len):
    """Yield complete histories that fit one last-mode request."""
    for sample_index, sample in enumerate(samples):
        conversations = sample["conversations"]
        best_candidate = None
        for history_end in range(2, len(conversations) + 1, 2):
            history = copy.deepcopy(conversations[:history_end])
            probe = history + [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": ""},
            ]
            prompt_length = _sharegpt_prompt_length(tokenizer, probe)
            if prompt_length >= input_len:
                break
            if prompt_length >= prefix_len:
                best_candidate = {
                    "source_index": sample_index,
                    "source_id": sample.get("id", f"sharegpt-{sample_index}"),
                    "history_turns": history_end // 2,
                    "history": history,
                    "probe_tokens": _sharegpt_prompt_token_ids(tokenizer, probe),
                    "prefix_base_tokens": prompt_length,
                }
                break
        if best_candidate is not None:
            yield best_candidate


def _sharegpt_prefix_candidates(tokenizer, samples, prefix_len, input_len):
    """Return all complete histories that fit one last-mode request."""
    return list(
        _iter_sharegpt_prefix_candidates(tokenizer, samples, prefix_len, input_len)
    )


def _select_prefix_groups(candidates, prefix_groups, prefix_len):
    """Choose histories that cannot share the configured cache prefix."""
    selected = []
    for candidate in candidates:
        if all(
            _common_prefix_length(
                [candidate["probe_tokens"], other["probe_tokens"]]
            ) < prefix_len
            for other in selected
        ):
            selected.append(candidate)
        if len(selected) == prefix_groups:
            return selected
    raise ValueError(
        "ShareGPT source dataset does not contain enough distinct complete "
        f"histories for {prefix_groups} prefix groups at prefix_len={prefix_len}"
    )


def _packed_history_prefix_probe(tokenizer, history):
    """Measure the prefix before request-specific final-user content."""
    probes = []
    for marker in ("0", "1"):
        conversations = copy.deepcopy(history)
        conversations.extend(
            [
                {"from": "human", "value": marker},
                {"from": "gpt", "value": ""},
            ]
        )
        probes.append(_sharegpt_prompt_token_ids(tokenizer, conversations))
    return _common_prefix_length(probes), probes[0]


def _packed_source_history(tokenizer, conversations, input_len):
    """Return the longest complete source history that leaves suffix room."""
    shared_length, probe_tokens = _packed_history_prefix_probe(
        tokenizer, conversations
    )
    if shared_length < input_len:
        return conversations, shared_length, probe_tokens

    best = None
    for history_end in range(2, len(conversations) + 1, 2):
        history = conversations[:history_end]
        shared_length, probe_tokens = _packed_history_prefix_probe(
            tokenizer, history
        )
        if shared_length >= input_len:
            break
        best = (history, shared_length, probe_tokens)
    return best


def _packed_prefix_units(tokenizer, samples, prefix_len, input_len, prefix_groups, min_units=None):
    """Tokenize a longest-first source pool large enough for all groups."""
    ranked_samples = sorted(
        enumerate(samples),
        key=lambda item: sum(
            len(message["value"]) for message in item[1]["conversations"]
        ),
        reverse=True,
    )
    empty_prefix_length, _ = _packed_history_prefix_probe(tokenizer, [])
    required_capacity = prefix_len * prefix_groups
    target_capacity = (required_capacity * 5 + 3) // 4
    effective_floor = prefix_groups if min_units is None else min_units
    minimum_units = max(prefix_groups * 4, effective_floor)
    capacity = 0
    units = []

    for sample_index, sample in ranked_samples:
        source_history = _packed_source_history(
            tokenizer, sample["conversations"], input_len
        )
        if source_history is None:
            continue
        history, shared_length, probe_tokens = source_history
        contribution = max(1, shared_length - empty_prefix_length)
        units.append(
            {
                "source_index": sample_index,
                "source_id": sample.get("id", f"sharegpt-{sample_index}"),
                "history": history,
                "history_turns": len(history) // 2,
                "probe_tokens": probe_tokens,
                "prefix_base_tokens": shared_length,
                "contribution_tokens": contribution,
            }
        )
        capacity += min(contribution, prefix_len)
        if len(units) >= minimum_units and capacity >= target_capacity:
            break

    if len(units) < prefix_groups or capacity < required_capacity:
        raise ValueError(
            "ShareGPT source dataset does not contain enough aggregate complete "
            f"history for {prefix_groups} packed prefix groups at "
            f"prefix_len={prefix_len}"
        )
    return units


def _select_packed_anchors(units, prefix_groups, prefix_len):
    """Reserve one longest, distinct under-target anchor for every group."""
    ordered = sorted(
        units,
        key=lambda unit: (
            0 if unit["prefix_base_tokens"] <= prefix_len else 1,
            abs(prefix_len - unit["prefix_base_tokens"]),
        ),
    )
    selected = []
    for unit in ordered:
        if all(
            _common_prefix_length([unit["probe_tokens"], other["probe_tokens"]])
            < prefix_len
            for other in selected
        ):
            selected.append(unit)
        if len(selected) == prefix_groups:
            return selected
    raise ValueError(
        "ShareGPT source dataset does not contain enough distinct anchors for "
        f"{prefix_groups} packed prefix groups at prefix_len={prefix_len}"
    )


def _packed_unit_order(units, current_length, prefix_len, input_len):
    # Prefer a unit that lands the running total inside the success window
    # [prefix_len, input_len): it reaches the target immediately without
    # overshooting input_len. Otherwise prefer the closest below-prefix unit
    # (grow toward target); units that overshoot input_len are last.
    def sort_key(unit):
        estimated_length = current_length + unit["contribution_tokens"]
        if prefix_len <= estimated_length < input_len:
            return 0, estimated_length - prefix_len
        if estimated_length <= prefix_len:
            return 1, prefix_len - estimated_length
        return 2, estimated_length - input_len

    return sorted(units, key=sort_key)


def _best_packed_fragment(tokenizer, history, units, prefix_len, input_len):
    """Use complete turns from one final source when its full history is too large."""
    below_target = []
    reached_target = []
    for unit in units:
        for history_end in range(2, len(unit["history"]) + 1, 2):
            combined = history + unit["history"][:history_end]
            shared_length, _ = _packed_history_prefix_probe(tokenizer, combined)
            if shared_length >= input_len:
                break
            candidate = (shared_length, unit, history_end)
            if shared_length >= prefix_len:
                reached_target.append(candidate)
                break
            below_target.append(candidate)

    if reached_target:
        return min(reached_target, key=lambda item: item[0] - prefix_len)
    if below_target:
        return max(below_target, key=lambda item: item[0])
    return None


def _fit_packed_message_prefix(tokenizer, history, message_index, prefix_len):
    """Truncate one shared message to the smallest prefix at or above target."""
    original_text = history[message_index]["value"]
    source_token_ids = tokenizer.encode(original_text, add_special_tokens=False)
    cache = {}

    def candidate_for(size):
        if size not in cache:
            history[message_index]["value"] = _decode_token_ids(
                tokenizer, source_token_ids[:size]
            )
            cache[size] = _packed_history_prefix_probe(tokenizer, history)
        return cache[size]

    empty_length, _ = candidate_for(0)
    full_length, _ = candidate_for(len(source_token_ids))
    if empty_length > prefix_len or full_length < prefix_len:
        history[message_index]["value"] = original_text
        return None

    guessed_size = max(
        0,
        min(
            len(source_token_ids),
            len(source_token_ids) - (full_length - prefix_len),
        ),
    )
    for _ in range(8):
        guessed_length, guessed_probe = candidate_for(guessed_size)
        if guessed_length == prefix_len:
            history[message_index]["value"] = _decode_token_ids(
                tokenizer, source_token_ids[:guessed_size]
            )
            return guessed_length, guessed_probe
        next_size = max(
            0,
            min(
                len(source_token_ids),
                guessed_size + (prefix_len - guessed_length),
            ),
        )
        if next_size == guessed_size:
            break
        guessed_size = next_size

    low = 0
    high = len(source_token_ids)
    while low < high:
        middle = (low + high) // 2
        candidate_length, _ = candidate_for(middle)
        if candidate_length < prefix_len:
            low = middle + 1
        else:
            high = middle

    candidate_sizes = {0, len(source_token_ids), low}
    candidate_sizes.update(
        range(max(0, low - 32), min(len(source_token_ids), low + 32) + 1)
    )
    fitted = []
    for size in sorted(candidate_sizes):
        candidate_length, probe_tokens = candidate_for(size)
        if candidate_length >= prefix_len:
            fitted.append((candidate_length, size, probe_tokens))
    if not fitted:
        history[message_index]["value"] = original_text
        return None

    fitted_length, fitted_size, probe_tokens = min(
        fitted, key=lambda item: (item[0] - prefix_len, item[1])
    )
    history[message_index]["value"] = _decode_token_ids(
        tokenizer, source_token_ids[:fitted_size]
    )
    return fitted_length, probe_tokens


def _fit_packed_history_prefix(
    tokenizer, history, source_units, prefix_len, input_len
):
    """Remove excess packed tail content and fit the shared prefix boundary."""
    history = copy.deepcopy(history)
    source_units = list(source_units)
    tail_truncated = False

    while len(history) >= 2:
        shorter_history = history[:-2]
        shorter_length, _ = _packed_history_prefix_probe(
            tokenizer, shorter_history
        )
        if shorter_length < prefix_len:
            break
        history = shorter_history
        tail_truncated = True
        source_unit, source_turns = source_units[-1]
        if source_turns == 1:
            source_units.pop()
        else:
            source_units[-1] = (source_unit, source_turns - 1)

    shared_length, probe_tokens = _packed_history_prefix_probe(tokenizer, history)
    if shared_length > prefix_len:
        fitted = _fit_packed_message_prefix(
            tokenizer, history, -1, prefix_len
        )
        if fitted is None:
            history[-1]["value"] = ""
            fitted = _fit_packed_message_prefix(
                tokenizer, history, -2, prefix_len
            )
        if fitted is None:
            raise RuntimeError(
                "ShareGPT packed prefix tail cannot be token-fitted to "
                f"prefix_len={prefix_len}"
            )
        shared_length, probe_tokens = fitted
        tail_truncated = True

    if shared_length < prefix_len or shared_length >= input_len:
        raise RuntimeError(
            "ShareGPT packed prefix validation failed after token fitting: "
            f"prefix={shared_length}, requested={prefix_len}, input_len={input_len}"
        )
    return history, source_units, shared_length, probe_tokens, tail_truncated


def _build_packed_prefix_groups(
    tokenizer, samples, prefix_len, input_len, prefix_groups, min_units=None
):
    units = _packed_prefix_units(
        tokenizer, samples, prefix_len, input_len, prefix_groups, min_units
    )
    anchors = _select_packed_anchors(units, prefix_groups, prefix_len)
    anchor_indices = {anchor["source_index"] for anchor in anchors}
    available = [
        unit for unit in units if unit["source_index"] not in anchor_indices
    ]
    groups = []

    for anchor in anchors:
        history = copy.deepcopy(anchor["history"])
        shared_length, probe_tokens = _packed_history_prefix_probe(
            tokenizer, history
        )
        source_units = [(anchor, anchor["history_turns"])]

        while shared_length < prefix_len:
            selected = None
            for unit in _packed_unit_order(available, shared_length, prefix_len, input_len):
                combined = history + unit["history"]
                candidate_length, candidate_probe = _packed_history_prefix_probe(
                    tokenizer, combined
                )
                if candidate_length < input_len:
                    selected = (unit, candidate_length, candidate_probe)
                    break

            if selected is not None:
                unit, shared_length, probe_tokens = selected
                history.extend(copy.deepcopy(unit["history"]))
                source_units.append((unit, unit["history_turns"]))
                available.remove(unit)
                continue

            fragment = _best_packed_fragment(
                tokenizer, history, available, prefix_len, input_len
            )
            if fragment is None:
                raise ValueError(
                    "ShareGPT packed prefix cannot reach prefix_len without "
                    f"exceeding input_len={input_len}"
                )
            shared_length, unit, history_end = fragment
            history.extend(copy.deepcopy(unit["history"][:history_end]))
            source_units.append((unit, history_end // 2))
            available.remove(unit)
            shared_length, probe_tokens = _packed_history_prefix_probe(
                tokenizer, history
            )

        (
            history,
            source_units,
            shared_length,
            probe_tokens,
            tail_truncated,
        ) = _fit_packed_history_prefix(
            tokenizer, history, source_units, prefix_len, input_len
        )
        source_ids = [unit["source_id"] for unit, _ in source_units]
        source_indices = [unit["source_index"] for unit, _ in source_units]
        source_turns = [turns for _, turns in source_units]
        groups.append(
            {
                "source_index": anchor["source_index"],
                "source_id": anchor["source_id"],
                "history_turns": len(history) // 2,
                "history": history,
                "probe_tokens": probe_tokens,
                "prefix_base_tokens": shared_length,
                "prefix_assembly": "packed",
                "source_ids": source_ids,
                "source_indices": source_indices,
                "source_turns": source_turns,
                "tail_truncated": tail_truncated,
            }
        )
    return groups


def _ordered_suffix_sources(pair_sources, group_candidate):
    """Prefer later turns from the prefix dialogue before using global donors."""
    if group_candidate.get("prefix_assembly") == "packed":
        prefix_sources = set(group_candidate["source_indices"])
        other = [
            source
            for source in pair_sources
            if source["source_index"] not in prefix_sources
        ]
        packed = [
            source
            for source in pair_sources
            if source["source_index"] in prefix_sources
        ]
        return other + packed

    own_source_index = group_candidate["source_index"]
    own_history_turns = group_candidate["history_turns"]
    related = [
        source
        for source in pair_sources
        if source["source_index"] == own_source_index
        and source["turn_index"] >= own_history_turns
    ]
    other = [
        source
        for source in pair_sources
        if source["source_index"] != own_source_index
    ]
    return related + other


def _fit_packed_last_human_content(tokenizer, conversations, target_len):
    """Fit a packed request directly, with the existing fitter as fallback."""
    original_text = conversations[-2]["value"]
    conversations[-2]["value"] = ""
    empty_prompt_len = _sharegpt_prompt_length(tokenizer, conversations)
    if empty_prompt_len > target_len:
        conversations[-2]["value"] = original_text
        raise ValueError(
            f"packed suffix leaves {empty_prompt_len} prompt tokens, above "
            f"input_len={target_len}"
        )
    if empty_prompt_len == target_len:
        return

    source_token_ids = tokenizer.encode(original_text, add_special_tokens=False)
    if not source_token_ids:
        source_token_ids = tokenizer.encode(
            "ShareGPT conversation", add_special_tokens=False
        )
    guessed_size = target_len - empty_prompt_len
    for _ in range(8):
        conversations[-2]["value"] = _decode_token_ids(
            tokenizer, _repeat_token_prefix(source_token_ids, guessed_size)
        )
        prompt_len = _sharegpt_prompt_length(tokenizer, conversations)
        if prompt_len == target_len:
            return
        next_size = max(0, guessed_size + target_len - prompt_len)
        if next_size == guessed_size:
            break
        guessed_size = next_size

    conversations[-2]["value"] = original_text
    _fit_last_human_content(tokenizer, conversations, target_len)


def _packed_suffix_units(
    tokenizer, suffix_sources, minimum_units, max_contribution
):
    pool_size = max(128, minimum_units * 8)

    def content_length(source):
        return len(source["question"]) + len(source["answer"])

    longest_sources = heapq.nlargest(
        pool_size // 2, suffix_sources, key=content_length
    )
    budget_sources = heapq.nlargest(
        pool_size,
        (
            source
            for source in suffix_sources
            if content_length(source) <= max_contribution
        ),
        key=content_length,
    )
    selected_sources = []
    selected_keys = set()
    for source in longest_sources + budget_sources:
        source_key = (source["source_index"], source["turn_index"])
        if source_key not in selected_keys:
            selected_keys.add(source_key)
            selected_sources.append(source)
    empty_prompt_len = _sharegpt_prompt_length(
        tokenizer,
        [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}],
    )
    units = []
    for source in selected_sources:
        history = [
            {"from": "human", "value": source["question"]},
            {"from": "gpt", "value": source["answer"]},
        ]
        prompt_len = _sharegpt_prompt_length(
            tokenizer,
            history
            + [
                {"from": "human", "value": ""},
                {"from": "gpt", "value": ""},
            ],
        )
        units.append(
            {
                "source": source,
                "history": history,
                "contribution_tokens": max(1, prompt_len - empty_prompt_len),
            }
        )
    return units


def _build_packed_dataset_prefix_entry(
    tokenizer, group_candidate, suffix_units, entry_offset, input_len
):
    conversations = copy.deepcopy(group_candidate["history"])
    used_units = []

    empty_final_pair = [
        {"from": "human", "value": ""},
        {"from": "gpt", "value": ""},
    ]
    current_len = _sharegpt_prompt_length(
        tokenizer, conversations + empty_final_pair
    )

    fitting_units = [
        unit
        for unit in suffix_units
        if current_len + unit["contribution_tokens"] < input_len
    ]
    oversized_units = [unit for unit in suffix_units if unit not in fitting_units]
    anchor_units = fitting_units + oversized_units
    forced_unit = anchor_units[entry_offset]
    ordered_units = [forced_unit] + [
        unit for unit in suffix_units if unit is not forced_unit
    ]

    if forced_unit in oversized_units:
        conversations.extend(copy.deepcopy(forced_unit["history"]))
        _fit_packed_last_human_content(tokenizer, conversations, input_len)
        if _sharegpt_prompt_length(tokenizer, conversations) != input_len:
            raise RuntimeError(
                f"ShareGPT packed suffix validation failed: expected {input_len}"
            )
        return {"conversations": conversations}, [forced_unit]

    conversations.extend(copy.deepcopy(forced_unit["history"]))
    used_units.append(forced_unit)
    current_len = _sharegpt_prompt_length(
        tokenizer, conversations + empty_final_pair
    )

    remaining_units = [
        unit for unit in ordered_units if unit is not forced_unit
    ]
    while len(used_units) < 64:
        available = [
            unit
            for unit in remaining_units
            if unit not in used_units
            and current_len + unit["contribution_tokens"] < input_len
        ]
        if not available:
            break
        selected = max(available, key=lambda unit: unit["contribution_tokens"])
        conversations.extend(copy.deepcopy(selected["history"]))
        used_units.append(selected)
        current_len = _sharegpt_prompt_length(
            tokenizer, conversations + empty_final_pair
        )

    final_unit = next(
        (unit for unit in ordered_units if unit not in used_units), forced_unit
    )
    conversations.extend(copy.deepcopy(final_unit["history"]))
    _fit_packed_last_human_content(tokenizer, conversations, input_len)
    if _sharegpt_prompt_length(tokenizer, conversations) != input_len:
        raise RuntimeError(
            f"ShareGPT packed suffix validation failed: expected {input_len}"
        )
    return {"conversations": conversations}, used_units + [final_unit]


def _build_dataset_prefix_entry(tokenizer, group_candidate, suffix_source, input_len):
    conversations = copy.deepcopy(group_candidate["history"])
    conversations.extend(
        [
            {"from": "human", "value": suffix_source["question"]},
            {"from": "gpt", "value": suffix_source["answer"]},
        ]
    )
    sample = {"conversations": conversations}
    _fit_last_human_content(tokenizer, sample["conversations"], input_len)
    if _sharegpt_prompt_length(tokenizer, sample["conversations"]) != input_len:
        raise RuntimeError(
            f"ShareGPT prefix-cache prompt length validation failed: expected "
            f"{input_len}"
        )
    return sample


def process_sharegpt_prefix_cache_dataset(
    tokenizer,
    source_path,
    output_path,
    input_len,
    prefix_len,
    prefix_groups,
    num_prompts,
    seed_requests=True,
    prefix_order="grouped",
    prefix_source="dataset",
    prefix_assembly="single",
    prefix_min_units=None,
):
    """Generate fixed-length last-mode requests with ShareGPT history prefixes.

    The default single assembly preserves one source conversation per prefix.
    Packed assembly combines long complete histories and request-specific suffix
    turns, then token-fits only the final shared and final-user messages.
    """
    for name, value in (
        ("input_len", input_len),
        ("prefix_len", prefix_len),
        ("prefix_groups", prefix_groups),
        ("num_prompts", num_prompts),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if prefix_len >= input_len:
        raise ValueError("prefix_len must be smaller than input_len")
    if prefix_groups > num_prompts:
        raise ValueError("prefix_groups cannot exceed num_prompts")
    if prefix_order not in {"grouped", "interleaved"}:
        raise ValueError("prefix_order must be grouped or interleaved")
    if prefix_source != "dataset":
        raise ValueError("prefix_source currently supports only 'dataset'")
    if prefix_assembly not in {"single", "packed"}:
        raise ValueError("prefix_assembly must be single or packed")

    with open(source_path, "r", encoding="utf-8") as source_file:
        source_dataset = json.load(source_file)
    if not isinstance(source_dataset, list):
        raise ValueError("ShareGPT source dataset must be a JSON array")
    valid_samples = [sample for sample in source_dataset if _validate_sharegpt_sample(sample)]
    if not valid_samples:
        raise ValueError("ShareGPT source dataset contains no valid conversations")
    pair_sources = _sharegpt_pair_sources(valid_samples)
    if not pair_sources:
        raise ValueError("ShareGPT source dataset contains no usable conversation pairs")
    if prefix_assembly == "packed":
        selected_groups = _build_packed_prefix_groups(
            tokenizer, valid_samples, prefix_len, input_len, prefix_groups, prefix_min_units
        )
    else:
        selected_groups = _select_prefix_groups(
            _iter_sharegpt_prefix_candidates(
                tokenizer, valid_samples, prefix_len, input_len
            ),
            prefix_groups,
            prefix_len,
        )

    base_count, remainder = divmod(num_prompts, prefix_groups)
    group_counts = [base_count + (index < remainder) for index in range(prefix_groups)]
    groups = []
    for group_id, (group_count, group_candidate) in enumerate(
        zip(group_counts, selected_groups)
    ):
        suffix_sources = _ordered_suffix_sources(pair_sources, group_candidate)
        if len(suffix_sources) < group_count:
            raise ValueError(
                "ShareGPT source dataset does not contain enough distinct final "
                f"turns for prefix group {group_id}"
            )
        suffix_units = None
        if prefix_assembly == "packed":
            suffix_units = _packed_suffix_units(
                tokenizer,
                suffix_sources,
                group_count,
                input_len - group_candidate["prefix_base_tokens"],
            )
        group_entries = []
        for offset in range(group_count):
            if prefix_assembly == "packed":
                entry, packed_suffix_units = _build_packed_dataset_prefix_entry(
                    tokenizer, group_candidate, suffix_units, offset, input_len
                )
                suffix_source = packed_suffix_units[-1]["source"]
            else:
                suffix_source = suffix_sources[offset]
                entry = _build_dataset_prefix_entry(
                    tokenizer, group_candidate, suffix_source, input_len
                )
            entry["id"] = (
                f"{group_candidate['source_id']}-s{group_candidate['source_index']}-"
                f"xtask-prefix-"
                f"{group_id}-{offset}"
            )
            entry["xtask_prefix_group"] = group_id
            entry["xtask_prefix_seed"] = bool(seed_requests and offset == 0)
            entry["xtask_prefix_source"] = prefix_source
            entry["xtask_prefix_history_source_id"] = group_candidate["source_id"]
            entry["xtask_prefix_history_source_index"] = group_candidate["source_index"]
            entry["xtask_prefix_history_turns"] = group_candidate["history_turns"]
            entry["xtask_prefix_suffix_source_id"] = suffix_source["source_id"]
            entry["xtask_prefix_suffix_source_index"] = suffix_source["source_index"]
            entry["xtask_prefix_suffix_turn"] = suffix_source["turn_index"]
            if prefix_assembly == "packed":
                entry["xtask_prefix_assembly"] = "packed"
                entry["xtask_prefix_history_source_ids"] = group_candidate[
                    "source_ids"
                ]
                entry["xtask_prefix_history_source_indices"] = group_candidate[
                    "source_indices"
                ]
                entry["xtask_prefix_history_source_turns"] = group_candidate[
                    "source_turns"
                ]
                entry["xtask_prefix_tail_truncated"] = group_candidate[
                    "tail_truncated"
                ]
                entry["xtask_prefix_suffix_source_ids"] = [
                    unit["source"]["source_id"] for unit in packed_suffix_units
                ]
                entry["xtask_prefix_suffix_source_indices"] = [
                    unit["source"]["source_index"]
                    for unit in packed_suffix_units
                ]
                entry["xtask_prefix_suffix_source_turns"] = [
                    unit["source"]["turn_index"] for unit in packed_suffix_units
                ]
            group_entries.append(entry)
        groups.append(group_entries)

    if seed_requests:
        generated = [group[0] for group in groups]
        remaining_groups = [group[1:] for group in groups]
    else:
        generated = []
        remaining_groups = groups

    if prefix_order == "grouped":
        generated.extend(entry for group in remaining_groups for entry in group)
    else:
        max_remaining = max((len(group) for group in remaining_groups), default=0)
        for offset in range(max_remaining):
            generated.extend(
                group[offset]
                for group in remaining_groups
                if offset < len(group)
            )

    token_lists_by_group = {
        group_id: [
            _sharegpt_prompt_token_ids(tokenizer, entry["conversations"])
            for entry in group
        ]
        for group_id, group in enumerate(groups)
    }
    for group_id, token_lists in token_lists_by_group.items():
        actual_shared_length = _common_prefix_length(token_lists)
        # The build measures the shared prefix with marker probes (a trailing
        # "0"/"1" human turn); the final requests use real suffixes whose
        # first token may merge across the boundary by a few tokens, so the
        # re-measured common prefix can land a little below the probe target.
        # Accept that small tokenizer-boundary deficit; a real shortfall is
        # still caught by the generous floor.
        if actual_shared_length < prefix_len - 32:
            raise RuntimeError(
                f"group {group_id} shared prefix is {actual_shared_length} tokens, "
                f"below requested {prefix_len}"
            )
        for entry in groups[group_id]:
            entry["xtask_prefix_tokens"] = actual_shared_length
    for left_group in range(prefix_groups):
        for right_group in range(left_group + 1, prefix_groups):
            cross_length = _common_prefix_length(
                [
                    token_lists_by_group[left_group][0],
                    token_lists_by_group[right_group][0],
                ]
            )
            if cross_length >= prefix_len:
                raise ValueError(
                    "prefix groups are not distinguishable before the requested "
                    f"prefix length ({prefix_len} tokens)"
                )

    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    temporary_path = f"{output_path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump(generated, output_file, ensure_ascii=False)
    os.replace(temporary_path, output_path)
    print(
        f"[process_dataset]: generated {len(generated)} ShareGPT prefix-cache "
        f"requests in {prefix_groups} groups: {output_path}"
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

    elif dataset_type == 'SHAREGPT_PREFIX_CACHE':
        if args.infermode.lower() != "last":
            raise ValueError(
                "ShareGPT prefix-cache preprocessing only supports infer_mode=last"
            )
        if not args.sourcepath:
            raise ValueError("--sourcepath is required for ShareGPT prefix-cache preprocessing")
        output_path = args.outputpath or os.path.join(
            data_path,
            f"ShareGPT-prefix-cache-i{input_len}-p{args.prefix_len}-"
            f"g{args.prefix_groups}-n{batch_size}.json",
        )
        process_sharegpt_prefix_cache_dataset(
            tokenizer,
            args.sourcepath,
            output_path,
            input_len=input_len,
            prefix_len=args.prefix_len,
            prefix_groups=args.prefix_groups,
            num_prompts=batch_size,
            seed_requests=bool(args.prefix_seed_requests),
            prefix_order=args.prefix_order,
            prefix_source=args.prefix_source,
            prefix_assembly=args.prefix_assembly,
            prefix_min_units=args.prefix_min_units,
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
