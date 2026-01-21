#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import csv
import os
import sys


# 第一个版本参数较少，仅对参数做简单暴力处理
# -i参数需要传入性能测试完成后输出的log文件路径，本脚本用于解析该日志结尾的指标表格
# 例如：'xx/xx/synthetic_results_4096_1024_2_ds-v32_GSM_0.log'
# -o参数需要传入一个路径，输出的summary.csv将会写入在这个路径中
# -a参数表示总卡数
# -d参数表示Decode卡数
try:
    input_idx = sys.argv.index('-i')
    input_log_dir = sys.argv[input_idx + 1]
except ValueError:
    input_log_dir = None  # 表示没有提供-i参数
output_idx = sys.argv.index('-o')
all_idx, decode_idx = sys.argv.index('-a'), sys.argv.index('-d')
info_idx = sys.argv.index('-f') if '-f' in sys.argv else 0

output_csv_dir = os.path.join(sys.argv[output_idx + 1], 'summary.csv')
all_cards_num, decode_cards_num = sys.argv[all_idx + 1], sys.argv[decode_idx + 1]
if info_idx > 0:
    test_info = sys.argv[info_idx + 1]
else:
    test_info = None
if not all_cards_num.isdigit() or not decode_cards_num.isdigit():
    raise ValueError('-d and -a should be positive integers.')
all_cards_num, decode_cards_num = int(all_cards_num), int(decode_cards_num)
if not all_cards_num or not decode_cards_num:
    raise ValueError('-d and -a should not be zero.')

# 此处识别对应测试的参数，如输入输出长度、并发等
# 由参数传入字符串：{input_length};{output_length};{request_count};{concurrency};{model_name};{dataset};{multimodal_test_file};{request_rate}
if test_info:
    test_info_parse_res = test_info.split(';')
    input_length, output_length, request_count, concurrency, model_name, dataset, multimodal_test_file, request_rate = test_info_parse_res
# 打屏：开始解析
print(f'[log_parsing]: 开始解析{input_log_dir}中的测试结果')


# 创建输出的summary.csv文件并写入表头
if not os.path.exists(output_csv_dir):
    csv_file = open(output_csv_dir, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'Dataset',
        'Model Name',
        'Multimodal_Test_File',
        'All Cards',
        'Decode Cards',
        'Input Length',
        'Output Length',
        'Request Count',
        'Concurrency Config',
        'Req Rate',
        'TTFT AVG(ms)',
        'TTFT P90(ms)',
        'TPOT AVG(ms)',
        'TPOT P90(ms)',
        'E2E AVG(s)',
        'Concurrency',
        'Total Token Throughput',
        'Total Token Throughput per Card',
        'Output Token Throughput',
        'Output Token Throughput per Card',
        'Output Decoding Token Throughput per Card',
        'QPS(req/s)',
        'QPM(req/min)'
    ])
    csv_file.close()
if input_log_dir is not None:
    with open(input_log_dir, encoding='utf-8') as f:
        res = f.readlines()

    summary = {
        'TTFT': {'Average': '', 'P90': ''},
        'TPOT': {'Average': '', 'P90': ''},
        'E2EL': {'Average': ''},
        'Concurrency': '',
        'Total Token Throughput': {'total': '', 'per_card':''},
        'Output Token Throughput': {'total':'', 'per_card': '', 'per_decoding_card': ''},
        'Request Throughput': {'qps': '', 'qpm': ''}
    }
    idx_avg, idx_p90, idx_val = 3, 8, 3

    # 读取log文件结尾的表格，首先记录各个关键列的索引，随后记录该列的值
    for i in res:
        if not i.startswith('│'):
            continue

        line = i.split('│')
        if line[1].strip() == 'Performance Parameters':
            for k, v in enumerate(line):
                if v.strip() == 'Average':
                    idx_avg = k
                elif v.strip() == 'P90':
                    idx_p90 = k
                    break
                else:
                    continue
        elif line[1].strip() == 'Common Metric':
            for k, v in enumerate(line):
                if v.strip() == 'Value':
                    idx_val = k
                    break
                else:
                    continue
        elif line[1].strip() == 'E2EL':
            e2e_time_in_ms = float(line[idx_avg].strip().split('.')[0])
            summary['E2EL']['Average'] = f'{e2e_time_in_ms / 1000} s'   # 此处换算时间单位为s
        elif line[1].strip() == 'TTFT':
            summary['TTFT']['Average'] = line[idx_avg].strip()
            summary['TTFT']['P90'] = line[idx_p90].strip()
        elif line[1].strip() == 'TPOT':
            summary['TPOT']['Average'] = line[idx_avg].strip()
            summary['TPOT']['P90'] = line[idx_p90].strip()
        elif line[1].strip() == 'Concurrency':
            summary['Concurrency'] = line[idx_val].strip()
        elif line[1].strip() == 'Request Throughput':
            qps = line[idx_val].strip().split()[0]
            qpm_value = float(qps) * 60
            summary['Request Throughput']['qps'] = qps
            summary['Request Throughput']['qpm'] = f"{qpm_value:.3f}"
        elif line[1].strip() == 'Output Token Throughput':
            output_throughput_total = line[idx_val].strip().split()[0]
            summary['Output Token Throughput']['total'] = output_throughput_total
            summary['Output Token Throughput']['per_card'] = str(float(output_throughput_total) / all_cards_num)
            summary['Output Token Throughput']['per_decoding_card'] = str(float(output_throughput_total) / decode_cards_num)
        elif line[1].strip() == 'Total Token Throughput':
            total_throughput_total = line[idx_val].strip().split()[0]
            summary['Total Token Throughput']['total'] = total_throughput_total
            summary['Total Token Throughput']['per_card'] = str(float(total_throughput_total) / all_cards_num)
        else:
            continue

    # 将需要统计的关键指标写入csv文件

    csv_file = open(output_csv_dir, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        dataset,
        model_name,
        multimodal_test_file,
        all_cards_num,
        decode_cards_num,
        input_length,
        output_length,
        request_count,
        concurrency,
        request_rate,
        summary['TTFT']['Average'].split()[0],
        summary['TTFT']['P90'].split()[0],
        summary['TPOT']['Average'].split()[0],
        summary['TPOT']['P90'].split()[0],
        summary['E2EL']['Average'].split()[0],
        summary['Concurrency'].split()[0],
        summary['Total Token Throughput']['total'],
        summary['Total Token Throughput']['per_card'],
        summary['Output Token Throughput']['total'],
        summary['Output Token Throughput']['per_card'],
        summary['Output Token Throughput']['per_decoding_card'],
        summary['Request Throughput']['qps'],
        summary['Request Throughput']['qpm']
])
    # 打屏：完成写入
    print(f'[log_parsing]: {input_log_dir}解析结果已写入{output_csv_dir}')


else:
    #无-i入参，只写入分隔符
    csv_file = open(output_csv_dir, 'a+', encoding='utf-8', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['---',] * 23)
csv_file.close()

