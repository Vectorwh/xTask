import os
import logging
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from enum import Enum


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    case_type: str = ""  # perf:性能, acc精度, all
    dataset_path: str = ""  # 数据集文件路径
    dataset_conf: str = ""  # 数据集目录下的配置文件，对应aisbench的datasets，其他bench工具可为空
    api_type: str = ""  # 对应服务化后端的类型 vllm_api_general_chat/vllm_api_general_stream
    num_prompts: str = ""
    max_out_len: str = ""
    batch_size: str = ""
    request_rate: str = ""
    baseline: str = ""
    threshold: float = 0.1
    max_num_worker: int = 1
    enable_debug: bool = False
    merge_ds: bool = False
    dump_eval_details: bool = False
    dump_extract_rate: bool = False
    pressure: bool = False


@dataclass
class TaskConfig:
    """任务配置基类"""
    # 基础配置
    case_name: str = ""
    model_path: str = ""
    served_model_name: str = ""
    start_device_id: int = 0
    max_memory_utiliization: float = 0.9
    nodes: int = 1
    world_size: int = 1
    host_addr: str = "0.0.0.0"
    port: int = 14830
    max_tokens_per_batch: int = 20000
    max_seqs_per_batch: int = 256
    enable_mla: bool = False
    block_size: int = 128
    communication_backend: str = "hccl"
    enable_prefix_cache: bool = False
    enable_chunked_prefill: bool = False
    enable_schedule_overlap: bool = False
    enable_mtp: bool = False
    dataset_name: str = "synthetic"   # synthetic: 随机数，GSM8K，sonnet
    dataset_path: str = ""
    dataset_format: str = "json"  # choices: ['json', 'jsonl', 'csv', 'huggingface']
    metrics: str = "perf"  # 支持的列表 e2el,ttft,tpot,itl,acc
    enhanced_result = False
    device_type = "A3"  # A2（8卡）,A2C(16卡),A3（8卡16die）
    serving_extensiable_param = ""
    additional_config: str = ""
    userid: int = os.getuid()

    # 基准测试配置
    benchmark_type: str = "aisbench"
    perf: BenchmarkConfig = field(default_factory=lambda: BenchmarkConfig(
        case_type="performance",
        api_type="vllm_api_general_stream",
        dataset_path="",
        batch_size=1,
        request_rate=0,
        max_out_len=1024,
        baseline="100",
        threshold=0.97,
        enable_debug=False,
        merge_ds=False,
        pressure=False
    ))
    acc: BenchmarkConfig = field(default_factory=lambda: BenchmarkConfig(
        case_type="accuracy",
        api_type="vllm_api_general_stream",
        dataset_path="",
        dataset_conf="",
        request_rate=0,
        max_out_len=1024,
        baseline="100",
        threshold=0.95,
        enable_debug=False,
        merge_ds=False,
        max_num_worker=1,
        dump_eval_details=False,
        dump_extract_rate=False
    ))
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskConfig':
        """从字典创建配置"""
        if data is None:
            data = {}
        
        # 提取基础配置
        config_data = {}
        for field_name in cls.__dataclass_fields__:
            if field_name not in ['perf', 'acc']:
                config_data[field_name] = data.get(field_name)
        
        # 处理benchmark部分
        benchmark_data = data.get('benchmark', {})
        config_data['benchmark_type'] = benchmark_data.get('benchmark_type', "vllm_ascend_bencmark")
        
        # 创建配置实例
        config = cls(**config_data)
        
        # 设置perf配置
        perf_data = benchmark_data.get('perf', {})
        if perf_data:
            config.perf = BenchmarkConfig(**perf_data)
        else:
            config.perf = BenchmarkConfig(case_type="performance", threshold=0.97)
        
        # 设置acc配置
        acc_data = benchmark_data.get('acc', {})
        if acc_data:
            config.acc = BenchmarkConfig(**acc_data)
        else:
            config.acc = BenchmarkConfig(
                case_type="accuracy",
                dataset_path="vllm",
                baseline="95",
                threshold=5
            )
        
        return config
    
    @classmethod
    def from_args(cls, args, xtask_config) -> 'TaskConfig':
        """ 从命令行参数创建配置"""
        config = cls()
        if hasattr(args, 'model_name'):
            setattr(config, "model_name", getattr(args, 'model_name', ""))
        
        # 设置基础配置
        setattr(config, "case_name", f"{args.model_name}_{args.framework}") 
        for field_name in cls.__dataclass_fields__:
            if field_name == "start_device_id":
                env_start_device_id = xtask_config.get("ENV_CONFIG").get("start_device_id")
                value = env_start_device_id if env_start_device_id else getattr(args, field_name)
                setattr(config, field_name, value)
                continue
    
            if field_name == "served_model_name":
                served_model_name = getattr(args, "model_path").split("/")[-1]
                value = served_model_name if served_model_name else getattr(args, field_name)
                setattr(config, field_name, value)
                continue


            if field_name not in ['perf', 'acc'] and hasattr(args, field_name):
                value = getattr(args, field_name)
                if value is not None:
                    setattr(config, field_name, value)
        
        # 设置benchmark_type
        if hasattr(args, 'benchmark_type'):
            config.benchmark_type = getattr(args, 'benchmark_type', "aisbench")
        
        # 设置perf配置
        if hasattr(args, 'perf_type'):
            config.perf.case_type = getattr(args, 'perf_case_type', "performance")
        if hasattr(args, 'dataset_path'):
            config.perf.dataset_path = getattr(args, 'dataset_path', "")
        if hasattr(args, 'dataset_conf'):
            config.perf.dataset_conf = getattr(args, 'dataset_conf', "")
        if hasattr(args, 'request_conf'):
            config.perf.request_conf = getattr(args, 'request_conf', "")
        if hasattr(args, 'num_prompts'):
            config.perf.num_prompts = getattr(args, 'num_prompts', "")
        if hasattr(args, 'max_out_len'):
            config.perf.max_out_len = getattr(args, 'max_out_len', "")
        if hasattr(args, 'batch_size'):
            config.perf.batch_size = getattr(args, 'batch_size', "")
        if hasattr(args, 'request_rate'):
            config.perf.request_rate = getattr(args, 'request_rate', "")
        if hasattr(args, 'perf_baseline'):
            config.perf.baseline = getattr(args, 'baseline', "")
        if hasattr(args, 'perf_threshold'):
            config.perf.threshold = getattr(args, 'perf_threshold', 0.05)
        
        # 设置acc配置
        if hasattr(args, 'acc_type'):
            config.acc.case_type = getattr(args, 'acc_case_type', "accuracy")
        if hasattr(args, 'dataset_path'):
            config.acc.dataset_path = getattr(args, 'dataset_path', "vllm")
        if hasattr(args, 'dataset_conf'):
            config.acc.dataset_conf = getattr(args, 'dataset_conf', "")
        if hasattr(args, 'request_conf'):
            config.acc.request_conf = getattr(args, 'request_conf', "")
        if hasattr(args, 'num_prompts'):
            config.acc.num_prompts = getattr(args, 'num_prompts', "")
        if hasattr(args, 'max_out_len'):
            config.acc.max_out_len = getattr(args, 'max_out_len', "")
        if hasattr(args, 'batch_size'):
            config.acc.batch_size = getattr(args, 'batch_size', "")
        if hasattr(args, 'request_rate'):
            config.acc.request_rate = getattr(args, 'request_rate', "")
        if hasattr(args, 'acc_baseline'):
            config.acc.baseline = getattr(args, 'acc_baseline', "95")
        if hasattr(args, 'acc_threshold'):
            config.acc.threshold = getattr(args, 'acc_threshold', 0.05)
        
        logging.info("Task info summary: %s"% config)

        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        
        # 重构为原始YAML结构
        result = {}
        benchmark_data = {
            'benchmark_type': data.pop('benchmark_type'),
            'perf': data.pop('perf'),
            'acc': data.pop('acc')
        }
        
        # 添加剩余的基础配置
        result.update(data)
        result['benchmark'] = benchmark_data
        
        return result
    
    def save_to_yaml(self, file_path: str) -> bool:
        """保存为YAML文件"""
        try:
            with open(file_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
            logging.info(f"Task config saved to: {file_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving task config to YAML: {e}")
            return False


class Task:
    """任务基类"""
    
    @staticmethod
    def load_from_yaml(file_path: str) -> Optional[TaskConfig]:
        """从YAML文件加载配置"""
        try:
            if not os.path.exists(file_path):
                logging.error(f"Config file not found: {file_path}")
                return None
            
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            return TaskConfig.from_dict(data)
        except Exception as e:
            logging.error(f"Error loading config from YAML: {e}")
            return None
    
    @staticmethod
    def create_from_template(template_name: str) -> Optional[TaskConfig]:
        """TODO 从模板创建配置"""
        try:
            template_path = os.path.join("templates", f"{template_name}.xlsx")
            return Task.load_from_yaml(template_path)
        except Exception as e:
            logging.error(f"Error creating task from template: {e}")
            return None
    
    @staticmethod
    def create_from_case(case_name: str) -> Optional[TaskConfig]:
        """从案例创建配置"""
        try:
            case_path = os.path.join("cases", f"{case_name}.yaml")
            return Task.load_from_yaml(case_path)
        except Exception as e:
            logging.error(f"Error creating task from case: {e}")
            return None
    
    @staticmethod
    def create_from_args(args, xtask_config) -> TaskConfig:
        """从命令行参数创建配置"""
        return TaskConfig.from_args(args, xtask_config)
