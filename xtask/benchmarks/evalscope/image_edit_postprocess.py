"""Qwen-Image-Edit（gedit）性能结果后处理。

复用 ``evalscope_postprocess`` 的解析链（find_evalscope_result_file /
EvalscopeResultParser.parse_perf_json / ResultSummary 结构完全同构），
仅附加两项注释性输出（不改变返回结构，不破坏现有消费方）：

1. 结果 metadata 追加 ``output_unit: 'MPx/s'`` 说明字段 —— gedit 插件
   把 completion_tokens 重定义为产出兆像素（MPx），因此
   ``output_tps_avg`` / ``Output Throughput (tok/s)`` 的量纲是 MPx/s；
2. ``ttft_avg`` 与 ``e2el_avg`` 差 < 5% 时打告警日志 —— 非流式出图
   场景 TTFT 即完整出图时延，二者应近似相等；若差异显著说明语义
   被误读（例如误按流式解读），提示人工核查。

用法与 ``evalscope_perf_post_process`` 完全一致：
    image_edit_perf_post_process(task_config, output_path, init_entries)
    -> (EvalscopeResult, benchmark_outputdir)
"""

import logging

from xtask.benchmarks.evalscope.evalscope_postprocess import (
    EvalscopeResultParser,
    find_evalscope_result_file,
)

# TTFT 与 E2EL 平均值差小于该比例时告警（非流式下二者应几乎相等）
_TTFT_E2EL_WARN_RATIO = 0.05


def image_edit_perf_post_process(task_config, output_path, init_entries):
    evalscope_output_file, benchmark_outputdir = find_evalscope_result_file(
        output_path, init_entries, evalscope_type="perf", dataset_name=""
    )
    if not evalscope_output_file:
        logging.error("image_edit perf的结果文件为空.")
        return None, benchmark_outputdir

    logging.info(
        f"image_edit perf的结果文件: {evalscope_output_file[0]}, {evalscope_output_file[1]}"
    )

    parser = EvalscopeResultParser()
    result = parser.parse_perf_json(evalscope_output_file)

    # ---- 附加 1：量纲注释字段（仅放 metadata，消费方按需读取） ----
    # gedit 插件重定义 completion_tokens = 产出 MPx，因此
    # output_tps_avg / summary_tps_total 的单位是 MPx/s（兆像素/秒）。
    result.metadata["output_unit"] = "MPx/s"
    result.metadata["output_unit_note"] = (
        "completion_tokens=retruned megapixels; "
        "output_tps_avg/output_tps_total/summary_tps_total are in MPx/s"
    )

    # ---- 附加 2：TTFT≈E2EL 校验（非流式：TTFT 即完整出图时延） ----
    summary = result.summary
    if summary is not None and summary.ttft_avg and summary.e2el_avg:
        diff_ratio = abs(summary.e2el_avg - summary.ttft_avg) / max(summary.e2el_avg, 1e-9)
        if diff_ratio < _TTFT_E2EL_WARN_RATIO:
            logging.info(
                "image_edit perf: ttft_avg(%.4f) 与 e2el_avg(%.4f) 差 %.2f%% < 5%%，"
                "符合非流式出图预期（TTFT=完整出图时延）",
                summary.ttft_avg,
                summary.e2el_avg,
                diff_ratio * 100,
            )
        else:
            logging.warning(
                "image_edit perf: ttft_avg(%.4f) 与 e2el_avg(%.4f) 差 %.2f%% >= 5%%，"
                "非流式出图场景二者应近似相等，请核查结果语义（TTFT=完整出图时延）",
                summary.ttft_avg,
                summary.e2el_avg,
                diff_ratio * 100,
            )

    # ---- 以下与 evalscope_perf_post_process 落盘行为一致 ----
    import json
    import os

    json_filename = task_config.get("case_name").lower() + ".json"
    json_path = os.path.join(output_path, json_filename)
    result.to_json(json_path)

    summary_filename = task_config.get("case_name").lower() + '_summary.txt'
    summary_path = os.path.join(output_path, summary_filename)
    summary_text = result.print_summary(detailed=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    logging.info("  ✅  结果已保存:")
    logging.info(f"   Result JSON file: {json_path}")
    logging.info(f"   Result Summary file: {summary_path}")

    return result, benchmark_outputdir
