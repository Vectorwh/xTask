"""xtask.plugins.evalscope.gedit_api 单元测试。

evalscope 依赖缺失时整体 skip（pytest.importorskip 保护）；
本机无 evalscope 时可用
PYTHONPATH=/tmp/evs14/unpacked 运行以测试真实插件行为。
"""

import base64
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path

import pytest

pytest.importorskip('evalscope.perf.arguments')

from evalscope.perf.arguments import Arguments

from xtask.plugins.evalscope.gedit_api import GEditImageGenApi


def make_tiny_png_bytes(width=4, height=3):
    """构造一个最小合法 PNG（无 PIL 依赖）。"""
    def chunk(tag, data):
        raw = tag + data
        return struct.pack('>I', len(data)) + raw + struct.pack('>I', _crc32(raw))

    def _crc32(data):
        # PNG 规定 CRC-32（多项式 0xEDB88320），无 zlib 时手算。
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                crc = (crc >> 1) ^ (0xEDB88320 if crc & 1 else 0)
        return crc ^ 0xFFFFFFFF

    ihdr = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0)
    raw_scanlines = b''.join(b'\x00' + b'\x11\x22\x33' * width for _ in range(height))
    return (
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', ihdr)
        + chunk(b'IDAT', _deflate_store(raw_scanlines))
        + chunk(b'IEND', b'')
    )


def _deflate_store(data):
    """最小 deflate（stored block）+ adler32，无 zlib 依赖。"""
    def adler32(buf):
        a, b = 1, 0
        for byte in buf:
            a = (a + byte) % 65521
            b = (b + a) % 65521
        return (b << 16) | a

    out = b'\x78\x01'
    for i in range(0, len(data), 65535):
        block = data[i:i + 65535]
        last = 1 if i + 65535 >= len(data) else 0
        out += bytes([last]) + struct.pack('<H', len(block)) + struct.pack('<H', len(block) ^ 0xFFFF)
        out += block
    out += struct.pack('>I', adler32(data))
    return out


class GeditApiTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arguments = Arguments(model='test-model', api='gedit_image_gen', url='http://127.0.0.1:8800/v1/image/generation')
        cls.plugin = GEditImageGenApi(cls.arguments)

    def _write_png(self, directory, name='a.png'):
        path = Path(directory) / name
        path.write_bytes(make_tiny_png_bytes())
        return str(path)


class BuildRequestTests(GeditApiTestBase):
    def test_build_request_converts_paths_to_b64(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = self._write_png(tmp)
            line = {
                'prompt': 'make it snowy',
                'images': [image_path],
                'size': '1328*1328',
                'num_inference_steps': 50,
                'guidance_scale': 3.5,
            }
            request = self.plugin.build_request(line)

            self.assertEqual(request['model'], 'test-model')
            self.assertEqual(request['input']['prompt'], 'make it snowy')
            encoded = request['input']['images'][0]
            self.assertEqual(base64.b64decode(encoded), Path(image_path).read_bytes())
            self.assertEqual(request['parameters']['size'], '1328*1328')
            self.assertEqual(request['parameters']['num_inference_steps'], 50)
            self.assertEqual(request['parameters']['guidance_scale'], 3.5)
            self.assertEqual(request['parameters']['seed'], 42)

    def test_build_request_seed_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = self._write_png(tmp)
            line = {'prompt': 'p', 'images': [image_path], 'seed': 7}
            self.assertEqual(self.plugin.build_request(line)['parameters']['seed'], 7)

            arguments = Arguments(
                model='test-model', api='gedit_image_gen',
                url='http://127.0.0.1:8800/v1/image/generation', seed=9,
            )
            plugin = GEditImageGenApi(arguments)
            line2 = {'prompt': 'p', 'images': [image_path]}
            self.assertEqual(plugin.build_request(line2)['parameters']['seed'], 9)
            # 无数据集行 seed、无 Arguments.seed -> 缺省 42
            self.assertEqual(self.plugin.build_request(line2)['parameters']['seed'], 42)

    def test_build_request_rejects_bad_lines(self):
        self.assertIsNone(self.plugin.build_request('plain string'))
        self.assertIsNone(self.plugin.build_request({'images': []}))
        self.assertIsNone(self.plugin.build_request({'prompt': 'p', 'images': ['/no/such/file.png']}))


class ParseResponsesTests(GeditApiTestBase):
    def _request_json(self, n_images=1):
        return json.dumps({
            'model': 'm',
            'input': {'prompt': 'p', 'images': ['x'] * n_images},
            'parameters': {},
        })

    def test_parse_responses_normal(self):
        payload = {'output': {'results': [
            {'image': 'abc', 'width': 1328, 'height': 1328},
        ]}}
        prompt_tokens, completion_tokens = self.plugin.parse_responses(
            [payload], request=self._request_json(n_images=2)
        )
        self.assertEqual(prompt_tokens, 2)
        self.assertAlmostEqual(completion_tokens, 1328 * 1328 / 1e6, places=6)

    def test_parse_responses_multiple_results_sums_pixels(self):
        payload = {'output': {'results': [
            {'image': 'a', 'width': 1000, 'height': 1000},
            {'image': 'b', 'width': 2000, 'height': 500},
        ]}}
        _, completion_tokens = self.plugin.parse_responses([payload], request=self._request_json())
        self.assertAlmostEqual(completion_tokens, (1000 * 1000 + 2000 * 500) / 1e6, places=6)

    def test_parse_responses_missing_image_returns_zero(self):
        payload = {'output': {'results': [{'image': '', 'width': 10, 'height': 10}]}}
        self.assertEqual(self.plugin.parse_responses([payload], request=self._request_json()), (0, 0))

    def test_parse_responses_empty_and_invalid(self):
        self.assertEqual(self.plugin.parse_responses([], request=None), (0, 0))
        self.assertEqual(self.plugin.parse_responses([{'no': 'output'}], request=None), (0, 0))
        self.assertEqual(self.plugin.parse_responses(['not-a-dict'], request=None), (0, 0))

    def test_parse_responses_bad_request_json(self):
        payload = {'output': {'results': [{'image': 'a', 'width': 5, 'height': 5}]}}
        self.assertEqual(self.plugin.parse_responses([payload], request='{bad json'), (0, 0.000025))

    def test_parse_responses_accepts_sanitized_payload(self):
        # 裁剪后的 payload（image 为摘要）仍能解析出 width/height
        sanitized = GEditImageGenApi.sanitize_response_payload({
            'output': {'results': [{'image': 'a' * 100, 'width': 100, 'height': 100}]}
        })
        _, completion_tokens = self.plugin.parse_responses([sanitized], request=self._request_json())
        self.assertAlmostEqual(completion_tokens, 100 * 100 / 1e6, places=6)


class SanitizeTests(GeditApiTestBase):
    def test_sanitize_request_body_replaces_b64(self):
        body = {
            'model': 'm',
            'input': {'prompt': 'p', 'images': ['AAAABBBB', 'CCCC']},
            'parameters': {'seed': 1},
        }
        sanitized = GEditImageGenApi.sanitize_request_body(body)
        self.assertEqual(sanitized['input']['images'], ['<b64:8>', '<b64:4>'])
        # 原体不被修改
        self.assertEqual(body['input']['images'], ['AAAABBBB', 'CCCC'])

    def test_sanitize_response_payload_keeps_dimensions_only(self):
        payload = {
            'output': {'results': [
                {'image': 'x' * 500, 'width': 320, 'height': 240, 'extra': 'dropped'},
            ]},
            'request_id': 'keep-top-level',
        }
        sanitized = GEditImageGenApi.sanitize_response_payload(payload)
        item = sanitized['output']['results'][0]
        self.assertEqual(item, {'image': '<b64:500>', 'width': 320, 'height': 240})
        self.assertEqual(sanitized['request_id'], 'keep-top-level')
        # 原体不被修改
        self.assertEqual(payload['output']['results'][0]['extra'], 'dropped')

    def test_sanitize_non_dict_inputs(self):
        self.assertEqual(GEditImageGenApi.sanitize_response_payload('text'), 'text')
        self.assertEqual(GEditImageGenApi.sanitize_request_body({'model': 'm'}), {'model': 'm'})

    def test_validate_payload(self):
        good = {'output': {'results': [{'image': 'a', 'width': 1, 'height': 1}]}}
        self.assertIsNone(GEditImageGenApi._validate_payload([good]))
        self.assertIsNotNone(GEditImageGenApi._validate_payload([{'output': {}}]))
        self.assertIsNotNone(GEditImageGenApi._validate_payload([]))


class B64CacheTests(GeditApiTestBase):
    def test_load_b64_caches_and_reports_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            image_path = self._write_png(tmp)
            first = self.plugin._load_b64(image_path)
            second = self.plugin._load_b64(image_path)
            self.assertEqual(first, second)
            self.assertIn(os.path.abspath(image_path), self.plugin._b64_cache)
            self.assertIsNone(self.plugin._load_b64(os.path.join(tmp, 'missing.png')))


if __name__ == '__main__':
    unittest.main()
