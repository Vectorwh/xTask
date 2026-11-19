# Qwen-Image-Edit 两段式图片编辑 API（/v1/image/generation）模型配置。
#
# 被 37_single_aisbench_gedit_acc（AISBench gedit, LMM-as-Judge 精度）使用：
#   - type=GEditImageEditAPI 来自 xtask/benchmarks/aisbench/gedit_image_edit_api.py，
#     依赖 PYTHONPATH 含 xTask 仓库根（调度器生成的脚本会自动 export）；
#   - host_ip/host_port 指向 DiT 服务（脚本 sed 替换）；
#   - encoder_url 指向 Text Encoder（两段式阶段 1）；注意：mmengine
#     Config.fromfile 的受限 exec 会把模块名替换为 lazy-import 代理，
#     配置内不能调用 os.environ.get 等运行时函数，因此这里只写字面量，
#     环境变量 GEDIT_ENCODER_URL 的覆盖逻辑在 GEditImageEditAPI.__init__
#     中实现（模型配置值 > 环境变量 > 默认）；
#   - 其余 generation_kwargs 为 DiT 请求体参数。
from xtask.benchmarks.aisbench.gedit_image_edit_api import GEditImageEditAPI

models = [
    dict(
        attr="service",
        type=GEditImageEditAPI,
        abbr="gedit-image-edit",
        path="",
        model="Qwen-Image-Edit-2509",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        api_key="",
        host_ip="localhost",
        host_port=8080,
        url="",
        max_out_len=512,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            encoder_url="http://127.0.0.1:18007",
            size="1328*1328",
            num_inference_steps=50,
            guidance_scale=3.5,
            true_cfg_scale=4.0,
            seed=42,
            temperature=0.01,
            ignore_eos=False,
        ),
    )
]
