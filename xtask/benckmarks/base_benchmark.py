from dataclasses import dataclass

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    average: float
    min: float
    max: float
    median: float
    p75: float
    p90: float
    p99: float
    n: int


@dataclass
class ResultSummary:
    """结果摘要数据类"""
    total_requests: int
    concurrency: int
    max_concurrency: int
    request_throughput: float
    input_tokens_avg: float
    output_tokens_avg: float
    output_token_throughput: float
    ttft_avg: float
    ttft_p90: float
    tpot_avg: float
    tpot_p90: float
    e2el_avg: float
    e2el_p90: float
