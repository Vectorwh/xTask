import logging

def download_dataset(task_config):
    download_cmd = list()
    if task_config.get("benchmark").get("acc"):
        acc_dataset_len = task_config.get("benchmark").get("acc").get("acc_dataset_len")
    else:
        acc_dataset_len = 0
    if task_config.get("dataset_name").lower() == "gsm8k" and task_config.get("benchmark").get("benchmark_type") == "aisbench":
        download_cmd = aisbench_download_gsm8k(download_cmd, acc_dataset_len)
    elif task_config.get("dataset_name").lower() == "ceavl" and task_config.get("benchmark").get("benchmark_type") == "aisbench":
        download_cmd = aisbench_download_ceavl(download_cmd, acc_dataset_len)
    
    return download_cmd


def aisbench_download_gsm8k(download_cmd, cut_len=0):
    download_cmd.append('\necho "下载gsm8k dataset：$AIS_BENCH_PATH/datasets/gsm8"')
    download_cmd.append('if [ ! -d "$AIS_BENCH_PATH/datasets/gsm8k" ]; then')
    download_cmd.append('    cd $AIS_BENCH_PATH/datasets')
    download_cmd.append('else')
    download_cmd.append('    echo "$AIS_BENCH_PATH/datasets/gsm8k 已存在."')
    download_cmd.append('    cd $AIS_BENCH_PATH/datasets')
    download_cmd.append('    rm -rf $AIS_BENCH_PATH/datasets/gsm8k')
    download_cmd.append('fi')
    download_cmd.append('wget http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip')
    download_cmd.append('unzip gsm8k.zip')
    download_cmd.append('rm gsm8k.zip')
    download_cmd.append('cp $AIS_BENCH_PATH/datasets/gsm8k/test.jsonl $AIS_BENCH_PATH/datasets/gsm8k/test.jsonl_bak')
    if cut_len:
        logging.info("gsm8k demo cut len is: %s"% cut_len)
        download_cmd.append(f'head -n {cut_len} $AIS_BENCH_PATH/datasets/gsm8k/test.jsonl_bak > $AIS_BENCH_PATH/datasets/gsm8k/test.jsonl')
    else:
        logging.info("cut_len is: %s, skip to cut dataset gsm8k."% cut_len)

    return download_cmd
    

def aisbench_download_ceavl(download_cmd):
    download_cmd.append('\necho "下载ceval dataset：$AIS_BENCH_PATH/datasets/下载ceval"')
    download_cmd.append('if [ ! -d "$AIS_BENCH_PATH/datasets/ceval" ]; then')
    download_cmd.append('    cd $AIS_BENCH_PATH/datasets')
    download_cmd.append('    mkdir -p ceval/formal_ceval')
    download_cmd.append('    cd ceval/formal_ceval')
    download_cmd.append('    wget https://www.modelscope.cn/datasets/opencompass/ceval-exam/resolve/master/ceval-exam.zip')
    download_cmd.append('    unzip ceval-exam.zip')
    download_cmd.append('    rm ceval-exam.zip')
    download_cmd.append('else')
    download_cmd.append('    echo "$AIS_BENCH_PATH/datasets/ceval 已存在."')
    download_cmd.append('fi')

    return download_cmd

