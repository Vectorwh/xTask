"""Qwen-Image-Edit / JoyAI-Image-Edit 图片编辑生成 API 插件（evalscope perf，两段式协议）。

两段式协议（Qwen-Image-Edit-2509 / JoyAI-Image-Edit-Plus 解耦部署）：
    阶段 1（Text Encoder, POST <GEDIT_ENCODER_URL>/v1/embeddings）：
        payload = {
            "model": "text_encoder",
            "input": "",
            "encoding_format": "float",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": [
                    {"type": "text", "text": "Picture 1: "},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<图>"}},
                    {"type": "text", "text": <user_prompt>},
                ]},
            ],
        }
        响应 result["data"][0]["mm_embeddings"][0]["embedding"] 的
        contents.bytes_contents 是 base64 的 bfloat16 原始字节，shape 为
        [token, hidden]；解析后去掉前 skip_tokens 个模板前缀 token 得
        prompt embedding（Qwen skip 64 / JoyAI skip 34，语义一致：保留
        vision/text token）。
        负向 prompt 用 " "（一个空格）走同样流程。
        给 encoder 的图片先 PIL resize 到 1024x1024（内存操作，不落盘）。

    阶段 2（DiT, POST /v1/image/generation）：
        input.prompt_embed / negative_prompt_embed 是 FP32 张量
        （contents.fp32_contents 为 float 列表，json 序列化后 payload 数十
        MB 属正常）；
        - Qwen：input.image 为原图 base64（不 resize）；
        - JoyAI：input.media_sources 为多图 base64 列表（1-5 张，
          DiT 侧逐图 bucket resize）；出图分辨率传 parameters.size
          （数据集行或缺省 1328*1328，实测不给 size 会退化 512*512）。

服务端响应（非流式）：

    {"output": {"results": [{"image": "<base64>", "width": 1328, "height": 1328}]}}

指标语义（借用 evalscope 的 prompt/completion token 槽位）：
    - ``prompt_tokens``  := 输入图片张数（build_request 时的 images 数量）
    - ``completion_tokens`` := 产出兆像素 MPx = sum(width * height) / 1e6

因此 benchmark_summary.json 中的：
    - ``Avg Input Tokens``  -> 平均输入图片张数
    - ``Avg Output Tokens`` -> 平均每次请求产出 MPx
    - ``Output Throughput (tok/s)`` -> MPx/s（产出兆像素每秒）

数据集行格式（--dataset line_by_line，每行一个 JSON dict）：
    {"prompt": "...",
     "images": ["/abs/path/a.png", ...],   # 存绝对路径，不存 base64
     "size": "1328*1328",
     "num_inference_steps": 50,
     "guidance_scale": 3.5,
     "seed": 42}        # 可选；缺省时 build_request 注入 42

框架默认把完整请求+响应（含 base64 图片与数万 float 的 embedding）无截断
写入 SQLite benchmark_data.db，本插件在 process_request 返回前做裁剪：
    - 请求侧 input.images / input.image 的 base64 替换为 "<b64:NB>" 摘要
    - 请求侧 prompt_embed / negative_prompt_embed 的 contents 替换为
      "<fp32:N> 摘要（N=元素个数）
    - 响应侧 output.results 每项只保留 width/height + "<b64:NB>" 摘要
裁剪发生在 accumulator.update（其内部惰性调用 parse_responses/finalize）
与 insert_benchmark_data 之前，parse_responses 仍能从保留的 width/height
读出 MPx，输入张数则来自请求侧未裁剪前的计数。
"""

import base64
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from evalscope.perf.arguments import Arguments
from evalscope.perf.plugin.api.default_api import DefaultApiPlugin
from evalscope.perf.plugin.registry import register_api
from evalscope.perf.utils.benchmark_util import BenchmarkData
from evalscope.utils.logger import get_logger

logger = get_logger()

# 请求/响应中 base64 字段的占位摘要前缀
B64_PLACEHOLDER_PREFIX = '<b64:'
# fp32_contents 的占位摘要前缀
FP32_PLACEHOLDER_PREFIX = '<fp32:'
DEFAULT_SEED = 42

# encoder 服务地址：环境变量优先，缺省本机 18007
DEFAULT_ENCODER_URL = 'http://127.0.0.1:18007'
ENCODER_URL_ENV = 'GEDIT_ENCODER_URL'

# ---------------------------------------------------------------------------
# 模型族协议（两段式协议的模型相关差异点）
# ---------------------------------------------------------------------------
# 检测依据：model_path（或 encoder_url 对应部署）下 model_index.json 的
# _class_name；也可经数据集行 / generation_kwargs 显式传 model_family 覆盖。
#
# qwen（QwenImageEditPlusPipeline，Qwen-Image-Edit-2509，缺省/零影响）：
#   - text encoder 为 Qwen2.5-VL；skip tokens = 64（xllm
#     pipeline_qwenimage_edit_plus.h prompt_template_encode_start_idx_ = 64）
#   - system prompt 描述输入图特征 + 编辑意图
#   - DiT 参考图走 input.image 单图（历史字段），分辨率由 parameters.size
#     固定（1328*1328 基准）
#
# joyai（JoyImageEditPlusPipeline，JoyAI-Image-Edit-Plus）：
#   - text encoder 为 Qwen3-VL-8B；skip tokens = 34（xllm
#     pipeline_joyimage_edit_plus.h kPromptEmbeddingStartIndex = 34）
#   - system prompt 为 Qwen3-VL 官方描述式模板；服务端 chat template 渲染
#     system+user 头恰好 34 token，skip 后保留 vision/text token（语义与
#     Qwen 的 64 一致：跳过模板前缀，保留图像与指令 token）
#   - DiT 参考图走 input.media_sources 多图（1-5 张逐图 bucket resize），
#     出图分辨率传 parameters.size（缺省 1328*1328；实测不给 size 会
#     退化为 512*512）
# 仅匹配图片编辑族专有标记（lower 后子串匹配，不做分隔符归一化）：
# 裸 'joyai' 会误命中 joyai-asr-xxx 等其它 JoyAI 系列模型，故收窄。
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

# 两段式协议常量（缺省保持 Qwen 行为零变化）
SYSTEM_PROMPT = QWEN_SYSTEM_PROMPT
NEGATIVE_PROMPT = ' '
ENCODER_IMAGE_SIZE = 1024
SKIP_TOKENS = 64
JOYAI_SKIP_TOKENS = 34


def _read_model_index_class(model_path):
    """读取 model_path 下 model_index.json 的 _class_name（失败返回 None）。"""
    if not model_path:
        return None
    try:
        with open(os.path.join(model_path, 'model_index.json'), 'r', encoding='utf-8') as f:
            index = json.load(f)
        return str(index.get('_class_name') or '')
    except (OSError, ValueError):
        return None


def detect_model_family(model_name_or_path):
    """按模型名/路径推断协议族：``joyai`` 或 ``qwen``（缺省，零影响）。

    依据（任一命中即 joyai）：
      1. model_name_or_path 本身含 joyai-image-edit / joyimageeditplus
         等图片编辑族专有标记（大小写不敏感）；
      2. 同目录 model_index.json 的 _class_name 为 JoyImageEditPlusPipeline。
    其余情况（含 None / 空串 / 路径不可读）一律返回 qwen，保持既有行为。
    """
    text = str(model_name_or_path or '')
    lowered = text.lower()
    if any(marker in lowered for marker in JOYAI_MODEL_MARKERS):
        return 'joyai'
    class_name = _read_model_index_class(text)
    if class_name and class_name.lower() == 'joyimageeditpluspipeline':
        return 'joyai'
    return 'qwen'


# 进程内模型族检测缓存：键为检测输入（served_model_name 或模型路径）的
# 字符串，值为 'qwen' / 'joyai'。检测候选在进程生命周期内不变，缓存命中
# 与未命中返回完全一致（纯性能优化）；避免每次 build_request 都去 13GB
# 模型目录 open/json.load 一次 model_index.json。dict 的 get/set 本身原子，
# 无需加锁。
_FAMILY_CACHE: Dict[str, str] = {}


def _detect_family_cached(model_name_or_path) -> str:
    """``detect_model_family`` 的进程内记忆化包装（同输入同输出）。"""
    key = str(model_name_or_path or '')
    family = _FAMILY_CACHE.get(key)
    if family is not None:
        return family
    family = detect_model_family(key)
    _FAMILY_CACHE[key] = family
    return family


def resolve_model_family(messages, param=None):
    """解析一次请求的模型族（数据集行 > 模型名/路径 > 缺省 qwen）。

    数据集行可显式给 ``model_family: joyai|qwen`` 覆盖；否则用模型名/路径
    自动检测（joyai-image-edit/joyimageeditplus 关键字，或 model_index.json
    的 _class_name）。缺省 ``qwen`` 保证既有 Qwen 用例行为零变化。
    检测结果按输入串记忆化（``_detect_family_cached``），纯性能优化。
    """
    if isinstance(messages, dict):
        explicit = messages.get('model_family')
        if isinstance(explicit, str) and explicit.strip():
            family = explicit.strip().lower()
            if family in ('joyai', 'qwen'):
                return family
            logger.warning(
                f'GEditImageGenApi: unknown model_family {explicit!r}, fallback to detection'
            )
    model_names = []
    if isinstance(messages, dict):
        model_names.append(messages.get('model'))
    if param is not None:
        model_names.append(getattr(param, 'model', None))
    for name in model_names:
        if name:
            family = _detect_family_cached(name)
            if family != 'qwen':
                return family
    return 'qwen'


def _b64_summary(value: Any) -> str:
    """把 base64 字符串替换为 ``<b64:NB>`` 摘要，非字符串原样返回。"""
    if isinstance(value, str):
        return f'{B64_PLACEHOLDER_PREFIX}{len(value)}>'
    return value


class _EncoderClient:
    """两段式协议阶段 1：Text Encoder /v1/embeddings 的轻量同步客户端。

    仅在 build_request 阶段（请求生成期，非计时窗口内）调用，同步
    requests.Session 复用连接；线程安全（Session 按插件实例单线程使用，
    evalscope 串行调用 build_request）。
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or os.environ.get(ENCODER_URL_ENV) or DEFAULT_ENCODER_URL).rstrip('/')
        self._session = None  # lazy，避免 import 期建连

    @property
    def session(self):
        import requests

        if self._session is None:
            self._session = requests.Session()
        return self._session

    def embed(
        self,
        prompt: str,
        images_b64: List[str],
        skip_tokens: int = SKIP_TOKENS,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        """调 encoder 拿一条 prompt（含图）的 embedding，返回 np.ndarray 或 None。"""
        import numpy as np

        content_items: List[Dict[str, Any]] = []
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
                {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
                {'role': 'user', 'content': content_items},
            ],
        }
        url = f'{self.base_url}/v1/embeddings'
        try:
            resp = self.session.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:  # noqa: BLE001
            logger.error(f'GEdit encoder request failed ({url}): {e}')
            return None

        try:
            entry = result['data'][0]['mm_embeddings'][0]['embedding']
            bytes_data = base64.b64decode(entry['contents']['bytes_contents'])
            shape = list(entry['shape'])
        except (KeyError, IndexError, TypeError, ValueError) as e:
            logger.error(f'GEdit encoder response missing expected fields: {e}')
            return None

        embed = self._bf16_to_f32(bytes_data, shape)
        if embed is None:
            return None
        if skip_tokens > 0 and embed.shape[0] > skip_tokens:
            embed = embed[skip_tokens:]
        return embed

    @staticmethod
    def _bf16_to_f32(raw: bytes, shape: List[int]):
        """base64 解出的 bfloat16 原始字节 -> float32 ndarray。"""
        import numpy as np

        try:
            arr = np.frombuffer(raw, dtype=np.uint8)
            if arr.size % 2:
                logger.error(f'GEdit encoder bf16 payload size {arr.size} not even')
                return None
            u16 = (arr[1::2].astype(np.uint16) << 8) | arr[0::2].astype(np.uint16)
            bits = (u16.astype(np.uint32) << 16).astype(np.uint32)
            f32 = bits.view(np.float32)
            expected = int(np.prod(shape)) if shape else 0
            if expected and f32.size != expected:
                logger.error(f'GEdit encoder bf16 size {f32.size} != shape {shape}')
                return None
            return f32.reshape(shape)
        except ValueError as e:
            logger.error(f'GEdit encoder bf16 decode failed: {e}')
            return None


@register_api('gedit_image_gen')
class GEditImageGenApi(DefaultApiPlugin):
    """Qwen-Image-Edit ``/v1/image/generation`` API 插件（两段式）。

    继承 :class:`DefaultApiPlugin` 复用其 HTTP 发送/异常/计时逻辑
    （本服务非流式，走其中的 application/json 分支），再叠加：
    1. build_request：路径 -> base64 -> 调 encoder 拿 pos/neg embedding ->
       组装两段式 DiT payload；
    2. process_request：业务校验（output.results[].image 必须存在）
       + SQLite 落库前的裁剪；
    3. parse_responses：返回 (输入图片张数, 产出 MPx)。
    """

    def __init__(self, param: Arguments):
        super().__init__(param)
        # 进程内 base64 缓存：同一图片路径只读盘一次。
        # 键为绝对路径，值为 base64 字符串；读取失败缓存 None 以免反复报错。
        self._b64_cache: Dict[str, Optional[str]] = {}
        # 进程内 embedding 缓存：负向 prompt 的 embedding 只算一次。
        self._neg_embed_cache: Optional[Any] = None
        self._encoder = _EncoderClient()

    # ------------------------------------------------------------------
    # 图片加载
    # ------------------------------------------------------------------
    def _load_b64(self, path: str) -> Optional[str]:
        """读取图片文件并编码为 base64（带进程内缓存）。"""
        key = os.path.abspath(path)
        if key in self._b64_cache:
            return self._b64_cache[key]
        value = None
        try:
            with open(path, 'rb') as f:
                value = base64.b64encode(f.read()).decode('ascii')
        except OSError as e:
            logger.error(f'Failed to read image file {path}: {e}')
        self._b64_cache[key] = value
        return value

    @staticmethod
    def _encoder_image_b64(raw_b64: Optional[str]) -> Optional[str]:
        """DiT 用原图 base64 -> encoder 用 1024x1024 JPEG base64（内存操作）。"""
        if not raw_b64:
            return None
        import io

        import PIL.Image
        import PIL.ImageOps

        try:
            pil_image = PIL.ImageOps.exif_transpose(PIL.Image.open(io.BytesIO(base64.b64decode(raw_b64))))
            if pil_image.mode == 'RGBA':
                background = PIL.Image.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, (0, 0), mask=pil_image)
                pil_image = background
            elif pil_image.mode not in ('RGB',):
                pil_image = pil_image.convert('RGB')
            pil_image = pil_image.resize((ENCODER_IMAGE_SIZE, ENCODER_IMAGE_SIZE))
            buf = io.BytesIO()
            pil_image.save(buf, format='JPEG')
            return base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception as e:  # noqa: BLE001
            logger.error(f'Failed to resize image for encoder: {e}')
            return None

    # ------------------------------------------------------------------
    # 请求构造（两段式）
    # ------------------------------------------------------------------
    def build_request(
        self, messages: Union[List[Dict], str, Dict], param: Optional[Arguments] = None
    ) -> Optional[Dict]:
        """组装两段式 ``/v1/image/generation`` 请求体。

        Args:
            messages: line_by_line 数据集产出的一行（JSON dict），
                形如 {"prompt", "images"(路径列表), "size",
                "num_inference_steps", "guidance_scale", "seed"}。
                JoyAI 用例可加 "model_family": "joyai" 显式指定（或省略，
                由模型名/路径自动检测）。
            param: 可选覆盖的 Arguments（缺省用 self.param）。

        Returns:
            请求体 dict；messages 非法或图片读取/encoder 调用失败时返回
            None（框架对 None 跳过该条请求）。
        """
        param = param or self.param
        if not isinstance(messages, dict):
            # evalscope 的 connect_test 会用 'hello'（str）探活；此时返回一个
            # 不带图、不带 embedding 的最小可发送请求，让 HTTP 层能真实打
            # 到 DiT 完成连接测试（业务结果不计入指标）。
            prompt = messages if isinstance(messages, str) else 'hello'
            return {
                'model': getattr(param, 'model', None) or 'Qwen-Image-Edit-2509',
                'input': {'prompt': prompt, 'negative_prompt': NEGATIVE_PROMPT},
                'parameters': {'num_inference_steps': 1, 'seed': DEFAULT_SEED},
            }

        prompt = messages.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            logger.error('GEditImageGenApi.build_request: missing "prompt" in dataset line')
            return None

        image_paths = messages.get('images') or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if not isinstance(image_paths, list):
            logger.error('GEditImageGenApi.build_request: "images" must be a list of paths')
            return None

        images_b64: List[str] = []
        for path in image_paths:
            encoded = self._load_b64(str(path))
            if encoded is None:
                return None
            images_b64.append(encoded)

        # ---- 模型族协议分支（缺省 qwen，既有 Qwen 行为零变化）----
        # resolve_model_family：数据集行 model_family > 模型名/路径检测
        # （joyai-image-edit/joyimageeditplus 关键字或 model_index.json 的
        # JoyImageEditPlusPipeline）> 缺省 qwen。
        model = getattr(param, 'model', None) or getattr(self.param, 'model', None)
        family = resolve_model_family(messages, param=param)
        if family == 'joyai':
            skip_tokens = JOYAI_SKIP_TOKENS
            system_prompt = JOYAI_SYSTEM_PROMPT
        else:
            skip_tokens = SKIP_TOKENS
            system_prompt = SYSTEM_PROMPT

        # ---- 阶段 1：调 encoder 拿 pos / neg embedding ----
        pos_embed = self._embed_prompt(prompt, images_b64, skip_tokens, system_prompt)
        if pos_embed is None:
            return None
        neg_embed = self._negative_embed(images_b64, skip_tokens, system_prompt)
        if neg_embed is None:
            return None

        parameters: Dict[str, Any] = {
            'num_inference_steps': messages.get('num_inference_steps', 50),
            'guidance_scale': messages.get('guidance_scale', 3.5),
            'true_cfg_scale': messages.get('true_cfg_scale', 4.0),
            'num_images_per_prompt': messages.get('num_images_per_prompt', 1),
            'max_sequence_length': messages.get('max_sequence_length', 2048),
        }
        size = messages.get('size')
        if size:
            parameters['size'] = size
        # seed：数据集行 > 服务参数（Arguments.seed）> 缺省 42
        seed = messages.get('seed')
        if seed is None:
            seed = getattr(param, 'seed', None)
        if seed is None:
            seed = DEFAULT_SEED
        parameters['seed'] = seed

        input_section: Dict[str, Any] = {
            'prompt': prompt,
            'negative_prompt': NEGATIVE_PROMPT,
            'prompt_embed': self._make_tensor(pos_embed, 'prompt_embeds'),
            'negative_prompt_embed': self._make_tensor(neg_embed, 'negative_prompt_embeds'),
        }
        if images_b64:
            if family == 'joyai':
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

        return {
            'model': model,
            'input': input_section,
            'parameters': parameters,
        }

    def _embed_prompt(
        self,
        prompt: str,
        images_b64: List[str],
        skip_tokens: int = SKIP_TOKENS,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        """正向 prompt embedding：多图时全部传给 encoder。"""
        encoder_images = []
        for b64_img in images_b64:
            resized = self._encoder_image_b64(b64_img)
            if resized is None:
                return None
            encoder_images.append(resized)
        return self._encoder.embed(prompt, encoder_images, skip_tokens=skip_tokens,
                                   system_prompt=system_prompt)

    def _negative_embed(
        self,
        images_b64: List[str],
        skip_tokens: int = SKIP_TOKENS,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        """负向 prompt embedding：进程内缓存（同一图片集合只算一次）。"""
        if self._neg_embed_cache is not None:
            return self._neg_embed_cache
        # 负向 embedding 与图相关（mm embedding 含图 token），按第一张图计算
        # 并缓存；性能用例数据集图片各异时收益有限但无正确性影响（负向
        # embedding 仅用作 CFG 条件）。
        encoder_images = []
        for b64_img in images_b64[:1]:
            resized = self._encoder_image_b64(b64_img)
            if resized is None:
                return None
            encoder_images.append(resized)
        embed = self._encoder.embed(
            NEGATIVE_PROMPT, encoder_images, skip_tokens=skip_tokens,
            system_prompt=system_prompt,
        )
        if embed is not None:
            self._neg_embed_cache = embed
        return embed

    @staticmethod
    def _make_tensor(data, name: str, datatype: str = 'FP32') -> Dict:
        """ndarray -> 两段式协议的张量 dict（fp32_contents 为 float 列表）。"""
        if not hasattr(data, 'shape'):
            import numpy as np

            data = np.asarray(data)
        shape = [int(dim) for dim in data.shape]
        if any(dim <= 0 for dim in shape):
            raise ValueError(f"Tensor '{name}' has invalid shape: {shape}")
        return {
            'name': name,
            'datatype': datatype,
            'shape': shape,
            'contents': {'fp32_contents': data.flatten().tolist()},
        }

    # ------------------------------------------------------------------
    # 指标解析
    # ------------------------------------------------------------------
    def parse_responses(
        self, responses: List[Dict], request: str = None, **kwargs: Any
    ) -> Tuple[int, int]:
        """解析响应，返回 (输入图片张数, 产出兆像素 MPx)。

        Args:
            responses: 框架收集的响应消息列表（非流式时仅一个元素，
                即响应 payload dict，可能已被本插件裁剪过 base64）。
            request: 请求 JSON 字符串（用于读取输入图片张数）。

        Returns:
            (int, int): 输入图片张数、sum(width*height)/1e6。
            异常 / 空响应 / 缺 image 字段一律返回 (0, 0)。
        """
        try:
            if not responses:
                return 0, 0
            payload = responses[-1]
            if not isinstance(payload, dict):
                return 0, 0

            results = ((payload.get('output') or {}).get('results')) or []
            if not results:
                return 0, 0
            # 业务校验：任何一项缺 image（含空串）都视为无效产出
            total_pixels = 0.0
            for item in results:
                image = item.get('image')
                if not isinstance(image, str) or not image:
                    return 0, 0
                width = item.get('width')
                height = item.get('height')
                if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
                    return 0, 0
                total_pixels += float(width) * float(height)

            input_images = self._count_request_images(request)
            return input_images, round(total_pixels / 1e6, 6)
        except Exception as e:  # noqa: BLE001  与框架插件一致的宽松失败语义
            logger.error(f'GEditImageGenApi.parse_responses failed: {e}')
            return 0, 0

    @staticmethod
    def _count_request_images(request: Optional[str]) -> int:
        """从请求 JSON 字符串数出输入图片张数（解析失败返回 0）。

        Qwen 请求 input.image 是单图、input.images 是历史多图列表；
        JoyAI 请求 input.media_sources 是多图列表；三者任一存在即计入。
        """
        if not request:
            return 0
        try:
            body = json.loads(request)
            input_section = body.get('input') or {}
            images = input_section.get('images')
            if isinstance(images, list) and images:
                return len(images)
            media_sources = input_section.get('media_sources')
            if isinstance(media_sources, list) and media_sources:
                return len(media_sources)
            if isinstance(input_section.get('image'), str) and input_section['image']:
                return 1
            return 0
        except (json.JSONDecodeError, AttributeError):
            return 0

    # ------------------------------------------------------------------
    # 裁剪（纯函数，便于单测）
    # ------------------------------------------------------------------
    @staticmethod
    def sanitize_request_body(body: Dict) -> Dict:
        """返回请求体的浅拷贝，base64 / embedding 裁剪为摘要。

        - input.images / input.image 的 base64 -> "<b64:NB>"
        - input.media_sources 每项的 base64 -> "<b64:NB>"（JoyAI 多图）
        - prompt_embed / negative_prompt_embed 的 contents.fp32_contents
          -> "<fp32:N>"（N=元素个数）
        输入图片张数在裁剪前先记录进 ``input.image_count``，供
        parse_responses 在请求 JSON 上数张数时使用。
        """
        sanitized = dict(body)
        input_section = sanitized.get('input')
        if not isinstance(input_section, dict):
            return sanitized
        new_input = dict(input_section)

        images = new_input.get('images')
        if isinstance(images, list):
            new_input['images'] = [_b64_summary(item) for item in images]
        if isinstance(new_input.get('image'), str):
            new_input['image'] = _b64_summary(new_input['image'])
        media_sources = new_input.get('media_sources')
        if isinstance(media_sources, list):
            new_media_sources = []
            for item in media_sources:
                if isinstance(item, dict) and isinstance(item.get('base64'), str):
                    trimmed = dict(item)
                    trimmed['base64'] = _b64_summary(item['base64'])
                    new_media_sources.append(trimmed)
                else:
                    new_media_sources.append(item)
            new_input['media_sources'] = new_media_sources

        for tensor_key in ('prompt_embed', 'negative_prompt_embed'):
            tensor = new_input.get(tensor_key)
            if isinstance(tensor, dict) and isinstance(tensor.get('contents'), dict):
                new_tensor = dict(tensor)
                new_contents = dict(tensor['contents'])
                fp32 = new_contents.get('fp32_contents')
                if isinstance(fp32, list):
                    new_contents['fp32_contents'] = f'{FP32_PLACEHOLDER_PREFIX}{len(fp32)}>'
                new_tensor['contents'] = new_contents
                new_input[tensor_key] = new_tensor

        sanitized['input'] = new_input
        return sanitized

    @staticmethod
    def sanitize_response_payload(payload: Any) -> Any:
        """返回响应 payload 的浅拷贝，output.results 每项只留摘要元数据。

        非_dict 输入原样返回；results 中每项替换为
        ``{"image": "<b64:NB>", "width": w, "height": h}``，
        width/height 缺失时记为 None（parse_responses 会判无效）。
        """
        if not isinstance(payload, dict):
            return payload
        sanitized = dict(payload)
        output_section = sanitized.get('output')
        if not isinstance(output_section, dict):
            return sanitized
        new_output = dict(output_section)
        results = new_output.get('results')
        if isinstance(results, list):
            new_results = []
            for item in results:
                if not isinstance(item, dict):
                    new_results.append(item)
                    continue
                new_results.append(
                    {
                        'image': _b64_summary(item.get('image')),
                        'width': item.get('width'),
                        'height': item.get('height'),
                    }
                )
            new_output['results'] = new_results
        sanitized['output'] = new_output
        return sanitized

    # ------------------------------------------------------------------
    # 请求执行（业务校验 + 裁剪）
    # ------------------------------------------------------------------
    async def process_request(
        self, client_session, url: str, headers: Dict, body: Dict
    ) -> BenchmarkData:
        """发送请求，校验业务结果并裁剪落库数据。

        流程：
        1. 调用 ``super().process_request`` 完成真实 HTTP 收发与计时；
        2. HTTP 成功后做业务校验：``output.results`` 非空且每项都有
           image 字段，否则置 ``success=False`` 并写 error；
        3. 无论成败，裁剪 ``BenchmarkData.request`` 与
           ``response_messages`` 中的 base64 / embedding，避免 SQLite 落库膨胀。
        """
        output = await super().process_request(client_session, url, headers, body)

        if output.success:
            error = self._validate_payload(output.response_messages)
            if error:
                output.success = False
                output.error = error
                logger.error(f'GEditImageGenApi business validation failed: {error}')

        # 裁剪：request 是 JSON 字符串，response_messages 是 [payload] 列表
        if isinstance(output.request, str):
            try:
                parsed_request = json.loads(output.request)
                if isinstance(parsed_request, dict):
                    output.request = json.dumps(
                        self.sanitize_request_body(parsed_request), ensure_ascii=False
                    )
            except json.JSONDecodeError:
                pass
        output.response_messages = [
            self.sanitize_response_payload(message) for message in output.response_messages
        ]
        return output

    @staticmethod
    def _validate_payload(response_messages: List[Any]) -> Optional[str]:
        """校验响应业务有效性，返回错误描述（None 表示有效）。"""
        if not response_messages:
            return 'empty response'
        payload = response_messages[-1]
        if not isinstance(payload, dict):
            return f'response is not a JSON object: {type(payload).__name__}'
        results = (payload.get('output') or {}).get('results')
        if not isinstance(results, list) or not results:
            return 'response has no output.results'
        for index, item in enumerate(results):
            image = item.get('image') if isinstance(item, dict) else None
            if not isinstance(image, str) or not image:
                return f'output.results[{index}].image is missing or empty'
        return None
