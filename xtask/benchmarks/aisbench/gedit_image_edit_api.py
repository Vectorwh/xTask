"""AISBench 模型包装：Qwen-Image-Edit / JoyAI-Image-Edit 两段式 /v1/image/generation。

gedit（GEdit-Bench）精度评测需要能出图的模型包装（LMMGenInferencer 期望
output.content 里是 PIL.Image）。AISBench 自带的 API 包装都是 chat 类
（/v1/chat/completions），Qwen-Image-Edit-2509 / JoyAI-Image-Edit-Plus
解耦部署形态是：

    阶段 1  Text Encoder  POST <GEDIT_ENCODER_URL>/v1/embeddings
            （图片 resize 1024x1024，bfloat16 解码，skip 64/34 token）
    阶段 2  DiT           POST <url>/v1/image/generation
            （带 prompt_embed / negative_prompt_embed /
             image(Qwen 单图) / media_sources(JoyAI 多图)）

模型族协议（与 evalscope 侧 xtask/plugins/evalscope/gedit_api.py 对称）：
    qwen（缺省，零影响）：skip 64、Qwen2.5-VL system prompt、
        input.image 单图、size 固定分辨率；
    joyai：skip 34（xllm pipeline_joyimage_edit_plus.h
        kPromptEmbeddingStartIndex=34）、Qwen3-VL system prompt、
        input.media_sources 多图（1-5 张，DiT 侧 bucket resize）。
    检测依据：model（served_model_name）或 path 含 joyai-image-edit/
    joyimageeditplus 等图片编辑族专有标记，或 path 下 model_index.json 的
    _class_name 为 JoyImageEditPlusPipeline；generation_kwargs 可显式传
    model_family 覆盖。

本模块提供 ``GEditImageEditAPI``（BaseAPIModel 子类，is_chat_api=False）：
    - get_request_body：从 PromptList 抽出 text 指令与 image_url 图片，
      先调 encoder 拿 pos/neg embedding，再组装两段式 DiT 请求体；
    - parse_text_response：把 output.results[0].image（b64）解码成
      PIL.Image 写进 output.content（LMMOutput 的 HANDLER_MAP 会把
      Image 对象存成 png 并返回相对路径，judge 数据集按该路径回读）。

与 evalscope perf 侧的 ``xtask/plugins/evalscope/gedit_api.py`` 共享同一
两段式协议常量（system prompt / 负向 prompt / skip tokens），但独立实现
以适配 AISBench 的 Output/aiohttp 体系。

encoder 地址读取顺序：模型配置 generation_kwargs.encoder_url >
环境变量 GEDIT_ENCODER_URL > http://127.0.0.1:18007。

judge 回读路径兼容（GEDIT_JUDGE_IMAGE_MIRROR，默认开启）：AISBench
LMMOutput._handle_image 把图片存到 work_dir/gedit_out_file/ 并在
prediction 里记录相对路径 "gedit_out_file/image_xxx.png"，但 gedit
LLM-judge 数据集（ImgSCJDGDataset / ImgPQJDGDataset 的
_load_from_predictions）以 predictions/<model_abbr>/ 为 base 拼接该
相对路径回读图片。两个基准目录不一致时 judge 拿不到图片且不报错，
直接把相对路径字符串当作 base64 发给 judge 服务，服务返回纯文本错误
体，AISBench 解析 JSON 失败 -> MODEL-DATA-003（warmup 阶段即失败）。
本模块 import 时对 LMMOutput._handle_image 打补丁：原逻辑落盘后，把
同一张 png 镜像到 work_dir/predictions/<model_abbr>/gedit_out_file/，
使 judge 的回读路径命中。镜像失败仅记 warning 不影响主流程；设置环境
变量 GEDIT_JUDGE_IMAGE_MIRROR=0 可关闭补丁恢复 AISBench 原行为。
"""

import base64
import io
import json
import os
import shutil
import urllib
from typing import Dict, Optional, Union

from ais_bench.benchmark.models import BaseAPIModel
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.models.output import Output

PromptType = Union[PromptList, str]

# ---------------------------------------------------------------------------
# 模型族协议（与 xtask/plugins/evalscope/gedit_api.py 对称的分支）
# ---------------------------------------------------------------------------
# qwen（缺省，零影响）：skip 64（xllm pipeline_qwenimage_edit_plus.h
#   prompt_template_encode_start_idx_ = 64）、Qwen2.5-VL system prompt、
#   DiT 参考图走 input.image 单图。
# joyai：skip 34（xllm pipeline_joyimage_edit_plus.h
#   kPromptEmbeddingStartIndex = 34）、Qwen3-VL system prompt、DiT 参考
#   图走 input.media_sources 多图（1-5 张逐图 bucket resize）。
#
# 标记仅取图片编辑族专有串（lower 后子串匹配，不做分隔符归一化）：裸
# 'joyai' 会误命中 joyai-asr-xxx 等其它 JoyAI 系列模型，故收窄。
JOYAI_MODEL_MARKERS = (
    'joyai-image-edit',
    'joyaiimageedit',
    'joyimageedit',
    'joyimageeditplus',
)

QWEN_SYSTEM_PROMPT = (
    'Describe the key features of the input image (color, size, texture, objects, background), '
    'then explain how the user\'s text instruction should be applied or modify the image. '
    'Generate a new image that meets the user\'s requirement '
    'while maintaining consistency with the original input where appropriate.'
)
JOYAI_SYSTEM_PROMPT = (
    ' \nDescribe the image by detailing the color, shape, size, texture, '
    'quantity, text, spatial relationships of the objects and background:'
)

# 兼容旧引用（Qwen 缺省常量，行为零变化）
SYSTEM_PROMPT = QWEN_SYSTEM_PROMPT
NEGATIVE_PROMPT = ' '
ENCODER_IMAGE_SIZE = 1024
SKIP_TOKENS = 64
JOYAI_SKIP_TOKENS = 34
DEFAULT_ENCODER_URL = 'http://127.0.0.1:18007'
ENCODER_URL_ENV = 'GEDIT_ENCODER_URL'

# 请求体级参数缺省（与 evalscope 侧插件 / 36 号 perf 用例保持一致）
DEFAULT_GEN_PARAMS = {
    'size': '1328*1328',
    'num_inference_steps': 50,
    'guidance_scale': 3.5,
    'true_cfg_scale': 4.0,
    'num_images_per_prompt': 1,
    'seed': 42,
    'max_sequence_length': 2048,
}


def _read_model_index_class(model_path):
    """读取 model_path 下 model_index.json 的 _class_name（失败返回 None）。"""
    if not model_path:
        return None
    try:
        with open(os.path.join(str(model_path), 'model_index.json'), 'r', encoding='utf-8') as f:
            index = json.load(f)
        return str(index.get('_class_name') or '')
    except (OSError, ValueError):
        return None


def detect_model_family(model_name_or_path):
    """按模型名/路径推断协议族：``joyai`` 或 ``qwen``（缺省，零影响）。

    依据（任一命中即 joyai）：字符串含 joyai-image-edit /
    joyimageeditplus 等图片编辑族专有标记（大小写不敏感），或同目录
    model_index.json 的 _class_name 为 JoyImageEditPlusPipeline。
    其余一律返回 qwen，保持既有行为。
    """
    text = str(model_name_or_path or '')
    lowered = text.lower()
    if any(marker in lowered for marker in JOYAI_MODEL_MARKERS):
        return 'joyai'
    class_name = _read_model_index_class(text)
    if class_name and class_name.lower() == 'joyimageeditpluspipeline':
        return 'joyai'
    return 'qwen'


@MODELS.register_module()
class GEditImageEditAPI(BaseAPIModel):
    """Qwen-Image-Edit 两段式图片编辑 API 包装（/v1/image/generation）。

    Args:
        path (str): 模型路径（仅用于展示，请求用 model 名）。
        model (str): 请求体 model 字段（served_model_name）。
        stream (bool): 必须为 False（非流式）。
        max_out_len (int): 未用于出图（占位兼容 AISBench 参数面）。
        retry (int): 失败重试次数。
        api_key (str): API key（可空）。
        host_ip (str): DiT 服务地址。
        host_port (int): DiT 服务端口。
        url (str): 完整 URL（优先于 host_ip/host_port）。
        generation_kwargs (dict): 透传请求体参数（size /
            num_inference_steps / guidance_scale / true_cfg_scale / seed /
            encoder_url 等）。
    """

    is_api: bool = True
    is_chat_api: bool = False

    def __init__(
        self,
        path: str = "",
        model: str = "",
        stream: bool = False,
        max_out_len: int = 4096,
        retry: int = 2,
        api_key: str = "",
        host_ip: str = "localhost",
        host_port: int = 8080,
        url: str = "",
        trust_remote_code: bool = False,
        generation_kwargs: Optional[Dict] = None,
        meta_template: Optional[Dict] = None,
        enable_ssl: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            path=path,
            stream=False,
            max_out_len=max_out_len,
            retry=retry,
            api_key=api_key,
            host_ip=host_ip,
            host_port=host_port,
            url=url,
            generation_kwargs=generation_kwargs,
            meta_template=meta_template,
            enable_ssl=enable_ssl,
            verbose=verbose,
        )
        self.model = model or path
        self.url = self._get_url()
        self._gen_params = dict(DEFAULT_GEN_PARAMS)
        if generation_kwargs:
            for key in (
                'size',
                'num_inference_steps',
                'guidance_scale',
                'true_cfg_scale',
                'num_images_per_prompt',
                'seed',
                'max_sequence_length',
                'encoder_url',
                'model_family',
            ):
                if key in generation_kwargs:
                    self._gen_params[key] = generation_kwargs[key]
        # 模型族：generation_kwargs.model_family 显式指定 > 按 model/path
        # 自动检测（joyai-image-edit/joyimageeditplus 标记或 model_index.json）>
        # 缺省 qwen（既有 Qwen 用例行为零变化）。
        explicit_family = self._gen_params.get('model_family')
        if isinstance(explicit_family, str) and explicit_family.strip().lower() in ('joyai', 'qwen'):
            self._model_family = explicit_family.strip().lower()
        elif detect_model_family(self.model) == 'joyai' or detect_model_family(path) == 'joyai':
            # served_model_name 与 path 任一命中即 joyai（detect 永远非空，
            # 不能用 or 短路——那会让 path 检测被跳过）。
            self._model_family = 'joyai'
        else:
            self._model_family = 'qwen'
        if self._model_family == 'joyai':
            self._skip_tokens = JOYAI_SKIP_TOKENS
            self._system_prompt = JOYAI_SYSTEM_PROMPT
        else:
            self._skip_tokens = SKIP_TOKENS
            self._system_prompt = QWEN_SYSTEM_PROMPT
        self._encoder_url = (
            os.environ.get(ENCODER_URL_ENV)
            or self._gen_params.get('encoder_url')
            or DEFAULT_ENCODER_URL
        ).rstrip('/')
        # requests.Session 复用连接（aiohttp 主循环外同步调用，见
        # get_request_body——它在 asyncio 事件循环里执行，但 encoder 调用
        # 采用 aiohttp 同 post 之外的独立 session，阻塞时长 ~0.1s 可接受）
        import aiohttp

        self._encoder_session: Optional[aiohttp.ClientSession] = None
        self._neg_embed_cache = None

    def _get_url(self) -> str:
        endpoint = "v1/image/generation"
        return urllib.parse.urljoin(self.base_url, endpoint)

    # ------------------------------------------------------------------
    # 两段式：阶段 1 encoder
    # ------------------------------------------------------------------
    async def _call_encoder(self, prompt: str, images_b64) -> Optional[list]:
        """调 Text Encoder 拿 embedding（bfloat16 -> float 列表）。"""
        import numpy as np
        import aiohttp

        content_items = []
        for idx, b64_img in enumerate(images_b64):
            content_items.append({'type': 'text', 'text': f'Picture {idx + 1}: '})
            content_items.append({
                'type': 'image_url',
                'image_url': {'url': f'data:image/jpeg;base64,{b64_img}'},
            })
        content_items.append({'type': 'text', 'text': prompt})
        payload = {
            'model': 'text_encoder',
            'input': '',
            'encoding_format': 'float',
            'messages': [
                {'role': 'system', 'content': [{'type': 'text', 'text': self._system_prompt}]},
                {'role': 'user', 'content': content_items},
            ],
        }
        if self._encoder_session is None:
            self._encoder_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=300)
            )
        try:
            async with self._encoder_session.post(
                f'{self._encoder_url}/v1/embeddings', json=payload
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    self.logger.error(
                        f'GEdit encoder HTTP {resp.status}: {text[:200]}'
                    )
                    return None
                result = await resp.json()
        except Exception as e:  # noqa: BLE001
            self.logger.error(f'GEdit encoder request failed: {e}')
            return None

        try:
            entry = result['data'][0]['mm_embeddings'][0]['embedding']
            raw = base64.b64decode(entry['contents']['bytes_contents'])
            shape = list(entry['shape'])
        except (KeyError, IndexError, TypeError, ValueError) as e:
            self.logger.error(f'GEdit encoder response malformed: {e}')
            return None

        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size % 2:
            self.logger.error(f'GEdit encoder bf16 payload size {arr.size} not even')
            return None
        u16 = (arr[1::2].astype(np.uint16) << 8) | arr[0::2].astype(np.uint16)
        f32 = (u16.astype(np.uint32) << 16).view(np.float32)
        if f32.size != int(np.prod(shape)):
            self.logger.error(f'GEdit encoder bf16 size {f32.size} != shape {shape}')
            return None
        embed = f32.reshape(shape)
        if self._skip_tokens > 0 and embed.shape[0] > self._skip_tokens:
            embed = embed[self._skip_tokens:]
        return embed.flatten().tolist(), list(embed.shape)

    @staticmethod
    def _resize_for_encoder(b64_img: str) -> Optional[str]:
        """原图 b64 -> 1024x1024 JPEG b64（内存操作）。"""
        import PIL.Image
        import PIL.ImageOps

        try:
            image = PIL.ImageOps.exif_transpose(
                PIL.Image.open(io.BytesIO(base64.b64decode(b64_img)))
            )
            if image.mode == 'RGBA':
                background = PIL.Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, (0, 0), mask=image)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize((ENCODER_IMAGE_SIZE, ENCODER_IMAGE_SIZE))
            buf = io.BytesIO()
            image.save(buf, format='JPEG')
            return base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception as e:  # noqa: BLE001
            return None

    @staticmethod
    def _make_tensor(flat: list, shape: list, name: str) -> Dict:
        return {
            'name': name,
            'datatype': 'FP32',
            'shape': shape,
            'contents': {'fp32_contents': flat},
        }

    # ------------------------------------------------------------------
    # AISBench API 面板
    # ------------------------------------------------------------------
    async def get_request_body(
        self, input_data: PromptType, max_out_len: int, output: Output, **args
    ):
        """PromptList -> 两段式 DiT 请求体。"""
        import PIL.Image

        def _collect_from_mm(parts):
            """[{'type':'text',...},{'type':'image_url',...}] -> (text, [b64])"""
            text = ''
            imgs = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get('type') == 'text' and isinstance(part.get('text'), str):
                    text += part['text']
                elif part.get('type') == 'image_url':
                    url = part.get('image_url')
                    if isinstance(url, dict):
                        url = url.get('url', '')
                    if isinstance(url, str) and url.startswith('data:image'):
                        imgs.append(url.split(',', 1)[1])
            return text, imgs

        prompt_text = ''
        images_b64 = []
        if isinstance(input_data, str):
            prompt_text = input_data
        else:
            # 兼容三种 PromptList 形态：
            # A. meta_template 包装：{'role':..., 'prompt_mm': [...]}
            # B. meta_template 包装纯文本：{'role':..., 'prompt': '...'}
            # C. 扁平（无 meta_template，BaseAPIModel.parse_template 的直接
            #    输出）：item 即 part dict {'type':'text'|'image_url', ...}
            for item in input_data:
                if isinstance(item, str):
                    prompt_text += item
                    continue
                if not isinstance(item, dict):
                    continue
                prompt_mm = item.get('prompt_mm')
                if prompt_mm is not None:
                    if isinstance(prompt_mm, dict):
                        prompt_mm = [prompt_mm]
                    text, imgs = _collect_from_mm(prompt_mm)
                    prompt_text += text
                    images_b64.extend(imgs)
                elif isinstance(item.get('prompt'), str):
                    prompt_text += item['prompt']
                elif 'type' in item:
                    text, imgs = _collect_from_mm([item])
                    prompt_text += text
                    images_b64.extend(imgs)
        output.input = prompt_text
        if not prompt_text or not images_b64:
            output.success = False
            output.error_info = (
                'GEditImageEditAPI requires one text instruction and at least '
                f'one image; got text={bool(prompt_text)} images={len(images_b64)}'
            )
            return None

        # ---- 阶段 1：encoder ----
        encoder_images = []
        for b64_img in images_b64:
            resized = self._resize_for_encoder(b64_img)
            if resized is None:
                output.success = False
                output.error_info = 'failed to resize input image for encoder'
                return None
            encoder_images.append(resized)

        pos = await self._call_encoder(prompt_text, encoder_images)
        if pos is None:
            output.success = False
            output.error_info = 'text encoder call failed'
            return None
        if self._neg_embed_cache is None:
            neg = await self._call_encoder(NEGATIVE_PROMPT, encoder_images[:1])
            if neg is None:
                output.success = False
                output.error_info = 'text encoder (negative) call failed'
                return None
            self._neg_embed_cache = neg

        pos_flat, pos_shape = pos
        neg_flat, neg_shape = self._neg_embed_cache

        # ---- 阶段 2：DiT 请求体 ----
        parameters = {
            key: value for key, value in self._gen_params.items()
            if key not in ('encoder_url', 'model_family')
        }
        # JoyAI 也传 parameters.size（缺省 1328*1328）：实测（20261103）不传
        # size 走 bucket 会退化为 512*512 出图，SC/PQ 崩塌（2.45/6.95 ->
        # 0.85/5.05）；显式 size 下 DiT 稳定输出全分辨率。JoyAI 与 Qwen
        # 在此行为一致，无需按模型族区分。
        input_section = {
            'prompt': prompt_text,
            'negative_prompt': NEGATIVE_PROMPT,
            'prompt_embed': self._make_tensor(pos_flat, pos_shape, 'prompt_embeds'),
            'negative_prompt_embed': self._make_tensor(
                neg_flat, neg_shape, 'negative_prompt_embeds'
            ),
        }
        if images_b64:
            if self._model_family == 'joyai':
                # JoyAI DiT：input.media_sources 多图（xllm
                # JoyImageEditPlusPipeline 逐图 bucket resize，1-5 张），
                # name/modality 必填（DiTSourceDecoder 校验）。
                input_section['media_sources'] = [
                    {
                        'type': 'base64',
                        'name': f'input_{idx}',
                        'base64': b64,
                        'modality': 'image',
                    }
                    for idx, b64 in enumerate(images_b64)
                ]
            else:
                # Qwen DiT（既有行为）：input.image 单图
                input_section['image'] = images_b64[0]
        request_body = {
            'model': self.model,
            'input': input_section,
            'parameters': parameters,
        }
        return request_body

    async def parse_text_response(self, api_response: dict, output: Output):
        """DiT 响应 -> PIL.Image 写入 output.content。"""
        import PIL.Image

        results = ((api_response or {}).get('output') or {}).get('results') or []
        if not results:
            output.success = False
            output.error_info = 'response has no output.results'
            return
        image_b64 = results[0].get('image')
        if not isinstance(image_b64, str) or not image_b64:
            output.success = False
            output.error_info = 'output.results[0].image is missing'
            return
        try:
            image = PIL.Image.open(io.BytesIO(base64.b64decode(image_b64)))
            image.load()
        except Exception as e:  # noqa: BLE001
            output.success = False
            output.error_info = f'failed to decode result image: {e}'
            return
        # LMMOutput.content 是 list；Image 对象由 HANDLER_MAP 落盘为 png
        if hasattr(output, 'content') and isinstance(output.content, list):
            output.content = [image]
        else:
            output.content = image

    async def parse_stream_response(self, api_response: dict, output: Output):
        await self.parse_text_response(api_response, output)

    async def close(self):
        if self._encoder_session is not None:
            await self._encoder_session.close()
            self._encoder_session = None


# ----------------------------------------------------------------------
# judge 回读路径兼容补丁
# ----------------------------------------------------------------------
def _patch_lmm_output_image_mirroring() -> None:
    """给 LMMOutput._handle_image 附加 judge 目录镜像（幂等）。"""
    import logging

    from ais_bench.benchmark.models import output as ais_output_module

    original = ais_output_module.LMMOutput._handle_image
    if getattr(original, '_gedit_judge_mirror', False):
        return  # 已打补丁，避免重复包装

    mirror_logger = logging.getLogger('xtask.gedit_judge_mirror')

    def _handle_image_with_judge_mirror(self, save_dir: str, index: int) -> str:
        rel_path = original(self, save_dir, index)
        try:
            if os.environ.get('GEDIT_JUDGE_IMAGE_MIRROR', '1') in ('0', 'false', 'False'):
                return rel_path
            # save_dir 形如 <work_dir>/<data_abbr>_out_file；
            # judge 以 <work_dir>/predictions/<model_abbr>/ 为 base 回读
            # rel_path（"gedit_out_file/image_xxx.png"）。
            work_dir = os.path.dirname(os.path.abspath(str(save_dir)))
            predictions_dir = os.path.join(work_dir, 'predictions')
            if not os.path.isdir(predictions_dir):
                return rel_path
            src = os.path.join(os.path.abspath(str(save_dir)), os.path.basename(rel_path))
            if not os.path.isfile(src):
                return rel_path
            for entry in os.listdir(predictions_dir):
                model_dir = os.path.join(predictions_dir, entry)
                if not os.path.isdir(model_dir):
                    continue
                mirror_dir = os.path.join(model_dir, os.path.dirname(rel_path))
                os.makedirs(mirror_dir, exist_ok=True)
                mirror_path = os.path.join(mirror_dir, os.path.basename(rel_path))
                if not os.path.exists(mirror_path):
                    shutil.copyfile(src, mirror_path)
        except Exception as exc:  # noqa: BLE001
            mirror_logger.warning(
                'gedit judge image mirror failed (non-fatal): %s', exc
            )
        return rel_path

    _handle_image_with_judge_mirror._gedit_judge_mirror = True
    ais_output_module.LMMOutput._handle_image = _handle_image_with_judge_mirror


_patch_lmm_output_image_mirroring()
