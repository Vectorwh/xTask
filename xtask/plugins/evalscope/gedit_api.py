"""Qwen-Image-Edit 图片编辑生成 API 插件（evalscope perf，两段式协议）。

两段式协议（Qwen-Image-Edit-2509 解耦部署）：
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
        [token, hidden]；解析后去掉前 64 个 skip_tokens 得 prompt embedding。
        负向 prompt 用 " "（一个空格）走同样流程。
        给 encoder 的图片先 PIL resize 到 1024x1024（内存操作，不落盘）。

    阶段 2（DiT, POST /v1/image/generation）：
        input.prompt_embed / negative_prompt_embed 是 FP32 张量
        （contents.fp32_contents 为 float 列表，json 序列化后 payload 数十
        MB 属正常）；image 为原图 base64（不 resize）。

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

# 两段式协议常量
SYSTEM_PROMPT = (
    'Describe the key features of the input image (color, size, texture, objects, background), '
    'then explain how the user\'s text instruction should be applied or modify the image. '
    'Generate a new image that meets the user\'s requirement '
    'while maintaining consistency with the original input where appropriate.'
)
NEGATIVE_PROMPT = ' '
ENCODER_IMAGE_SIZE = 1024
SKIP_TOKENS = 64


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

    def embed(self, prompt: str, images_b64: List[str], skip_tokens: int = SKIP_TOKENS):
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
                {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
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

        # ---- 阶段 1：调 encoder 拿 pos / neg embedding ----
        pos_embed = self._embed_prompt(prompt, images_b64)
        if pos_embed is None:
            return None
        neg_embed = self._negative_embed(images_b64)
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

        model = getattr(param, 'model', None) or getattr(self.param, 'model', None)

        input_section: Dict[str, Any] = {
            'prompt': prompt,
            'negative_prompt': NEGATIVE_PROMPT,
            'prompt_embed': self._make_tensor(pos_embed, 'prompt_embeds'),
            'negative_prompt_embed': self._make_tensor(neg_embed, 'negative_prompt_embeds'),
        }
        if images_b64:
            input_section['image'] = images_b64[0]

        return {
            'model': model,
            'input': input_section,
            'parameters': parameters,
        }

    def _embed_prompt(self, prompt: str, images_b64: List[str]):
        """正向 prompt embedding：多图时全部传给 encoder。"""
        encoder_images = []
        for b64_img in images_b64:
            resized = self._encoder_image_b64(b64_img)
            if resized is None:
                return None
            encoder_images.append(resized)
        return self._encoder.embed(prompt, encoder_images)

    def _negative_embed(self, images_b64: List[str]):
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
        embed = self._encoder.embed(NEGATIVE_PROMPT, encoder_images)
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

        两段式请求 input.image 是单图，input.images 是历史多图列表；
        二者任一存在即计入。
        """
        if not request:
            return 0
        try:
            body = json.loads(request)
            input_section = body.get('input') or {}
            images = input_section.get('images')
            if isinstance(images, list) and images:
                return len(images)
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
