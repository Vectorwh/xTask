# framework/core/argument_checker.py
import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class ConfigSource(Enum):
    """Configuration source type"""
    TEMPLATE = "template"
    CASE_NAME = "case_name"
    ARGUMENTS = "arguments"
    UNKNOWN = "unknown"


@dataclass
class ValidationResult:
    """Validation result container"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    config_source: ConfigSource
    config_data: Dict[str, Any] = None


class ArgumentChecker:
    """Argument validation and configuration generator"""
    
    # Required argument groups for different configuration sources
    REQUIRED_GROUPS = {
        ConfigSource.TEMPLATE: ['template'],
        ConfigSource.CASE_NAME: ['case_name'],
        ConfigSource.ARGUMENTS: ['model_name', 'model_path']
    }
    
    # Optional arguments
    OPTIONAL_ARGS = [
        'model_name', 'model_path', 'framework',
        'dataset_name', 'dataset_path', 'dataset_format',
        'benchmark_type', 'benchmark_config',
        'port', 'host', 'gpu_ids',
        'input_length', 'output_length', 'batch_size',
        'container_image', 'container_runtime',
        'output_dir', 'log_dir', 'verbose'
    ]
    
    def __init__(self, config_dir: str = "config"):
        """Initialize argument checker with configuration directories"""
        self.config_dir = Path(config_dir)
        self.case_dir = self.config_dir / "cases"
        self.template_dir = self.config_dir / "templates"
        
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Model Inference Testing Framework",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Usage Examples:
  1. Using template file: python main.py --template templates/test_cases.xlsx
  2. Using case name: python main.py --case_name qwen3_8b_xllm_001
  3. Using direct arguments: python main.py --model_name gpt2 --model_path /models/gpt2 --benchmark_type lm-eval
            """
        )
        
        # Input source arguments (mutually exclusive)
        input_group = parser.add_mutually_exclusive_group(required=True)
        input_group.add_argument(
            "--template", "-t",
            type=str,
            help="Excel template file path containing multiple test cases"
        )
        input_group.add_argument(
            "--case_name", "-c",
            type=str,
            help="Test case name, corresponding to YAML file in config/cases/"
        )
        input_group.add_argument(
            "--model_name", "-m",
            type=str,
            help="Model name (direct argument mode)"
        )
        
        # Model parameters
        parser.add_argument(
            "--model_path", "-mp",
            type=str,
            help="Model path (local or remote URL)"
        )
        parser.add_argument(
            "--framework", "-fmk",
            type=str,
            choices=['xllm', 'mindie', 'vllm-ascend'],
            default='xllm',
            help="Inference framework type"
        )
        parser.add_argument(
            "--trust_remote_code", "-trc",
            type=bool,
            default=False,
            help="是否信任远程代码"
        )
        
        # Dataset parameters
        parser.add_argument(
            "--dataset_name", "-dn",
            type=str,
            help="Dataset name"
        )
        parser.add_argument(
            "--dataset_path", "-dp",
            type=str,
            help="Dataset path"
        )
        parser.add_argument(
            "--dataset_conf",
            type=str,
            help="Dataset config"
        )
        parser.add_argument(
            "--dataset_format", "-df",
            type=str,
            choices=['json', 'jsonl', 'csv', 'huggingface'],
            default='json',
            help="Dataset format"
        )
        parser.add_argument(
            "--block_size",
            type=int,
            default=128,
            help="block_size"
        )
        parser.add_argument(
            "--communication_backend",
            type=str,
            default="hccl",
            help="communication_backend"
        )
        parser.add_argument(
            "--enhanced_result",
            type=bool,
            default=False,
            help="additional onfig"
        )
        parser.add_argument(
            "--device_type",
            choices=['A2', 'A2C', 'A3'],
            type=str,
            default='A3',
            help="device type"
        )
        parser.add_argument(
            "--serving_extensiable_param",
            type=str,
            help="additional onfig"
        )
        parser.add_argument(
            "--additional_config",
            type=str,
            help="additional config"
        )
        
        # Benchmark parameters
        parser.add_argument(
            "--benchmark_type", "-bt",
            type=str,
            choices=['aisbench', 'vllm_bench', 'xllm_bench', 'jd_vllm_bench'],
            default='aisbench',
            help="Benchmark tool type"
        )
        parser.add_argument(
            "--benchmark_config", "-bc",
            type=str,
            help="Benchmark configuration file path"
        )
        parser.add_argument(
            "--request_conf",
            type=str,
            help="request conf"
        )
        parser.add_argument(
            "--request_rate",
            type=int,
            default=0,
            help="request rate"
        )
        parser.add_argument(
            "--perf_baseline",
            type=str,
            help="perf_baseline"
        )
        parser.add_argument(
            "--acc_baseline",
            type=str,
            help="request conf"
        )
        parser.add_argument(
            "--perf_threshold",
            type=float,
            default=0.05,
            help="perf_threshold"
        )
        parser.add_argument(
            "--acc_threshold",
            type=float,
            default=0.05,
            help="acc_threshold"
        )
        parser.add_argument(
            "--metrics",
            type=str,
            default='perf',
            help="benchmark metrics"
        )
        
        # Inference service parameters
        parser.add_argument(
            "--port", "-p",
            type=int,
            default=8000,
            help="Inference service port"
        )
        parser.add_argument(
            "--host", "-H",
            type=str,
            default="localhost",
            help="Inference service host address"
        )
        parser.add_argument(
            "--world_size",
            type=int,
            default=1,
            help="world_size"
        )
        parser.add_argument(
            "--nodes",
            type=int,
            default=1,
            help="nodes"
        )
        parser.add_argument(
            "--start_device_id",
            type=int,
            default=0,
            help="start_device_id"
        )
        parser.add_argument(
            "--gpu_ids", "-g",
            type=str,
            help="GPU ID list, comma-separated"
        )
        parser.add_argument(
            "--max_memory_utiliization",
            type=float,
            default=0.9,
            help="max_memory_utiliization"
        )
        
        # Test task parameters
        parser.add_argument(
            "--input_length",
            type=int,
            default=512,
            help="Input sequence length/num prompts"
        )
        parser.add_argument(
            "--output_length",
            type=int,
            default=128,
            help="Output sequence length/max_out_len"
        )
        parser.add_argument(
            "--batch_size", "-bs",
            type=int,
            default=1,
            help="Batch size"
        )
        parser.add_argument(
            "--max_seqs_per_batch",
            type=int,
            default=256,
            help="max_seqs_per_batch"
        )
        parser.add_argument(
            "--max_tokens_per_batch",
            type=int,
            default=1024,
            help="max_tokens_per_batch"
        )

        parser.add_argument(
            "--enable_prefix_cache",
            type=bool,
            default=False,
            help="enable_prefix_cache"
        )
        parser.add_argument(
            "--enable_chunked_prefill",
            type=bool,
            default=False,
            help="enable_chunked_prefill"
        )
        parser.add_argument(
            "--enable_schedule_overlap",
            type=bool,
            default=False,
            help="enable_schedule_overlap"
        )
        parser.add_argument(
            "--enable_mtp",
            type=bool,
            default=False,
            help="enable_mtp"
        )
        parser.add_argument(
            "--enable_mla",
            type=bool,
            default=False,
            help="enable_mla"
        )

        # Container parameters
        parser.add_argument(
            "--container_image", "-ci",
            type=str,
            help="Container image name"
        )
        parser.add_argument(
            "--container_runtime", "-cr",
            type=str,
            choices=['docker', 'podman', 'nerdctl'],
            default='docker',
            help="Container runtime"
        )
        
        # Output parameters
        parser.add_argument(
            "--output_dir", "-o",
            type=str,
            default="results",
            help="Results output directory"
        )
        parser.add_argument(
            "--log_dir", "-l",
            type=str,
            default="logs",
            help="Log directory"
        )
        
        # Other parameters
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Verbose output mode"
        )
        parser.add_argument(
            "--dry_run",
            action="store_true",
            help="Dry run mode, only check without execution"
        )
        
        return parser.parse_args()
    
    def validate_arguments(self, args: argparse.Namespace) -> ValidationResult:
        """Validate argument correctness and completeness"""
        errors = []
        warnings = []
        config_source = ConfigSource.UNKNOWN
        
        # Determine configuration source
        if args.template:
            config_source = ConfigSource.TEMPLATE
            if not self._validate_template(args.template):
                errors.append(f"Template file does not exist or has invalid format: {args.template}")
                
        elif args.case_name:
            config_source = ConfigSource.CASE_NAME
            if not self._validate_case_name(args.case_name):
                errors.append(f"Test case does not exist: {args.case_name}")
                
        elif args.model_name:
            config_source = ConfigSource.ARGUMENTS
            if not self._validate_direct_args(args):
                errors.extend(self._get_direct_args_errors(args))
        
        else:
            errors.append("Must provide one of: --template, --case_name, or --model_name")
        
        # Check required arguments for each configuration source
        required_args = self.REQUIRED_GROUPS.get(config_source, [])
        for arg in required_args:
            if not getattr(args, arg, None):
                errors.append(f"Argument '{arg}' is required in {config_source.value} mode")
        
        # Logical argument validation
        if config_source == ConfigSource.ARGUMENTS:
            # Validate model path
            if args.model_path and not self._validate_model_path(args.model_path):
                warnings.append(f"Model path may not exist: {args.model_path}")
            
            # Validate dataset path
            if args.dataset_path and not self._validate_dataset_path(args.dataset_path):
                warnings.append(f"Dataset path may not exist: {args.dataset_path}")
            
            # Validate port range
            if args.port < 1024 or args.port > 65535:
                errors.append(f"Port must be in range 1024-65535: {args.port}")
            
            # Validate GPU ID format
            if args.gpu_ids:
                if not self._validate_gpu_ids(args.gpu_ids):
                    errors.append(f"GPU ID format error, should be comma-separated numbers: {args.gpu_ids}")
        
        # Generate configuration data if no errors
        config_data = None
        if not errors:
            config_data = self._generate_config_data(args, config_source)
            
            # Validate generated configuration
            config_errors = self._validate_config_data(config_data)
            errors.extend(config_errors)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            config_source=config_source,
            config_data=config_data
        )
    
    def _validate_template(self, template_path: str) -> bool:
        """Validate template file existence and format"""
        path = Path(template_path)
        if not path.exists():
            return False
        
        # Check file format
        if path.suffix.lower() not in ['.xlsx', '.xls']:
            return False
            
        return True
    
    def _validate_case_name(self, case_name: str) -> bool:
        """Validate if test case exists"""
        case_file = self.case_dir / f"{case_name}.yaml"
        return case_file.exists()
    
    def _validate_direct_args(self, args: argparse.Namespace) -> bool:
        """Validate direct argument mode parameters"""
        # Check required arguments
        if not args.model_name or not args.model_path:
            return False
        
        # If benchmark_type specified, check benchmark_config
        if args.benchmark_type and args.benchmark_type != 'custom':
            default_config = self._get_default_benchmark_config(args.benchmark_type)
            if not default_config:
                return False
        
        return True
    
    def _get_direct_args_errors(self, args: argparse.Namespace) -> List[str]:
        """Get error messages for direct argument mode"""
        errors = []
        
        if not args.model_name:
            errors.append("model_name argument is required")
        if not args.model_path:
            errors.append("model_path argument is required")
        
        if args.benchmark_type and args.benchmark_type != 'custom':
            if not args.benchmark_config:
                default_config = self._get_default_benchmark_config(args.benchmark_type)
                if not default_config:
                    errors.append(f"benchmark_type '{args.benchmark_type}' requires benchmark_config specification")
        
        return errors
    
    def _validate_model_path(self, model_path: str) -> bool:
        """Validate model path existence or URL format"""
        path = Path(model_path)
        
        # Check if it's a URL
        if model_path.startswith(('http://', 'https://', 's3://', 'gs://')):
            return True
        
        # Check local path
        if path.exists():
            # Check if it's a directory with model files
            if path.is_dir():
                model_files = list(path.glob("*.bin")) + list(path.glob("*.pt")) + list(path.glob("*.safetensors"))
                return len(model_files) > 0
            return True
        
        return False
    
    def _validate_dataset_path(self, dataset_path: str) -> bool:
        """Validate dataset path existence or URL format"""
        path = Path(dataset_path)
        
        # Check if it's a URL or HuggingFace dataset
        if dataset_path.startswith(('http://', 'https://', 'hf://', 'huggingface://')):
            return True
        
        # Check local path
        if path.exists():
            return True
        
        return False
    
    def _validate_gpu_ids(self, gpu_ids: str) -> bool:
        """Validate GPU ID format"""
        try:
            ids = [int(id.strip()) for id in gpu_ids.split(',')]
            return all(id >= 0 for id in ids)
        except ValueError:
            return False
    
    def _validate_config_data(self, config_data: Dict) -> List[str]:
        """Validate configuration data structure and values"""
        errors = []
        
        # Basic structure validation
        required_sections = ['model', 'inference']
        for section in required_sections:
            if section not in config_data:
                errors.append(f"Configuration missing required section: {section}")
        
        # Model configuration validation
        if 'model' in config_data:
            model = config_data['model']
            if 'name' not in model:
                errors.append("Model configuration missing 'name' field")
            if 'path' not in model:
                errors.append("Model configuration missing 'path' field")
            if 'framework' not in model:
                errors.append("Model configuration missing 'framework' field")
        
        # Inference service validation
        if 'inference' in config_data:
            inference = config_data['inference']
            if 'port' not in inference:
                errors.append("Inference configuration missing 'port' field")
            else:
                port = inference['port']
                if not isinstance(port, int) or port < 1024 or port > 65535:
                    errors.append(f"Invalid port number: {port}")
        
        return errors
    
    def _get_default_benchmark_config(self, benchmark_type: str) -> Optional[str]:
        """Get default benchmark configuration file path"""
        config_map = {
            "benchmark_type": benchmark_type,
            "perf": {
                "baseline": '',
                "batch_size": '',
                "case_type": "performance",
                "dataset_conf": '',
                "dataset_path": '',
                "max_out_len": '',
                "num_prompts": '',
                "request_conf": '',
                "request_rate": '',
                "threshold": 0.97
            }
        }
        
        config_path = config_map.get(benchmark_type)
        if config_path and Path(config_path).exists():
            return config_path
        
        return config_map
    
    def _generate_config_data(self, args: argparse.Namespace, 
                             config_source: ConfigSource) -> Dict:
        """Generate configuration data from arguments"""
        
        if config_source == ConfigSource.CASE_NAME:
            # Load from case file
            return self._load_case_config(args.case_name)
        
        elif config_source == ConfigSource.TEMPLATE:
            # Return template file path for main program to process
            return {
                'template_path': args.template,
                'source': 'template'
            }
        
        else:  # ARGUMENTS
            # Build configuration dictionary
            config = {
                'model': {
                    'name': args.model_name,
                    'path': args.model_path,
                    'framework': args.framework,
                    'benchmark_type': args.benchmark_type
                },
                'inference': {
                    'port': args.port,
                    'host': args.host,
                    'container_runtime': args.container_runtime,
                    'gpu_ids': self._parse_gpu_ids(args.gpu_ids) if args.gpu_ids else []
                },
                'task': {
                    'input_length': args.input_length,
                    'output_length': args.output_length,
                    'batch_size': args.batch_size
                },
                'results': {
                    'output_dir': args.output_dir,
                    'save_format': 'json'
                }
            }
            
            # Add dataset configuration if provided
            if args.dataset_name or args.dataset_path:
                config['dataset'] = {
                    'name': args.dataset_name or 'custom_dataset',
                    'path': args.dataset_path,
                    'format': args.dataset_format
                }
            
            # Add benchmark configuration if provided
            if args.benchmark_type:
                config['benchmark'] = {
                    'type': args.benchmark_type,
                    'config': args.benchmark_config or 
                             self._get_default_benchmark_config(args.benchmark_type)
                }
            
            # Add container image configuration
            if args.container_image:
                config['inference']['container_image'] = args.container_image
            else:
                # Generate default container image name based on framework
                framework = args.framework
                config['inference']['container_image'] = \
                    f"inference-{framework}:latest"
            
            return config
    
    def _load_case_config(self, case_name: str) -> Dict:
        """Load configuration from test case YAML file"""
        case_file = self.case_dir / f"{case_name}.yaml"
        with open(case_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _parse_gpu_ids(self, gpu_ids_str: str) -> List[int]:
        """Parse comma-separated GPU ID string into list"""
        if not gpu_ids_str:
            return []
        
        return [int(id.strip()) for id in gpu_ids_str.split(',')]
    
    def print_validation_result(self, result: ValidationResult):
        """Print validation results in formatted output"""
        logging.info("=" * 60)
        logging.info("Argument Validation Result:")
        
        
        if result.is_valid:
            logging.info(f"  ✅  Validation PASSED (Source: {result.config_source.value})")
            
            if result.warnings:
                logging.info("  ⚠️  Warnings:")
                for warning in result.warnings:
                    logging.warning(f"  - {warning}")
            
            logging.info(f"  📋  Configuration Summary:")
            self._print_config_summary(result.config_data)
            
        else:
            logging.error("  ❌  Validation FAILED")
            logging.error("Error Details:")
            for error in result.errors:
                logging.error(f"  - {error}")
            
            if result.warnings:
                logging.info("  ⚠️  Warnings:")
                for warning in result.warnings:
                    logging.warning(f"  - {warning}")
            
            logging.info("  💡  Suggestions:")
            logging.info("  1. Check if all required arguments are provided")
            logging.info("  2. Verify file paths are correct")
            logging.info("  3. Use --help to view complete argument documentation")
        logging.info("=" * 60)
    
    def _print_config_summary(self, config: Dict):
        """Print configuration summary in readable format"""
        if config.get('source') == 'template':
            logging.info(f"  Template File: {config.get('template_path')}")
            return
        
        # Model information
        if 'model' in config:
            model = config['model']
            logging.info(f"  Model: {model.get('name')} ({model.get('framework')})")
            logging.info(f"  Path: {model.get('path')}")
        
        # Inference service
        if 'inference' in config:
            inference = config['inference']
            logging.info(f"  Service: {inference.get('host')}:{inference.get('port')}")
            if inference.get('gpu_ids'):
                logging.info(f"  GPU IDs: {inference.get('gpu_ids')}")
        
        # Dataset
        if 'dataset' in config:
            dataset = config['dataset']
            logging.info(f"  Dataset: {dataset.get('name')}")
        
        # Benchmark
        if 'benchmark' in config:
            benchmark = config['benchmark']
            logging.info(f"  Benchmark: {benchmark.get('type')}")


def main():
    """Main function for command line testing"""
    checker = ArgumentChecker()
    
    try:
        args = checker.parse_arguments()
        result = checker.validate_arguments(args)
        checker.print_validation_result(result)
        sys.exit(0 if result.is_valid else 1)
    except Exception as e:
        logging.error(f"  ❌  Argument validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
