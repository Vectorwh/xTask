import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

import aiohttp
from tqdm.asyncio import tqdm

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    best_of: int = 1
    use_beam_search: bool = False


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0
    ttft: float = 0  # Time to first token
    itl: List[float] = field(
        default_factory=list)  # List of inter-token latencies
    prompt_len: int = 0
    error: str = ""

async def async_request_npu(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        params = {
            "max_new_tokens": request_func_input.output_len,
            "details": False,
            "do_sample": True,
            "repetition_penalty": 1,
            "seed": 123,
            "temperature": 1,
            "top_k":50
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,headers=headers) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode(),
                                              "data: ")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft
                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
                else:
                    print("occury_error,http status != 200")
                    print("error input:",json.dumps(payload))
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue

                        chunk = remove_prefix(chunk_bytes.decode(),
                                              "data: ")

                        data = json.loads(chunk)
                        print(data)                
        except Exception:    
            print("llc_occury_error")
            output.success = False
            exc_info = sys.exc_info()
              
            output.error = "".join(traceback.format_exception(*exc_info))
            print(output.error)
        if pbar:
            pbar.update(1)
        return output



async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        params = {
            "best_of": request_func_input.best_of,
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = remove_prefix(chunk.decode("utf-8"), "data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        assert request_func_input.best_of == 1
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = remove_prefix(chunk.decode("utf-8"), "data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp -
                                              most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.generated_text = json.loads(data)["text_output"]
                    output.success = True

                else:
                    output.error = response.reason
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii( request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert request_func_input.best_of == 1
        assert not request_func_input.use_beam_search

        payload = {
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(url=request_func_input.api_url,
                                    json=payload) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    output.generated_text = parsed_resp["text"][0]
                    output.success = True
                else:
                    output.error = response.reason
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "OpenAI Completions API URL must end with 'completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = remove_prefix(chunk.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            data = json.loads(chunk)

                            if data["choices"][0]["text"]:
                                timestamp = time.perf_counter()
                                # First token
                                if ttft == 0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                # NOTE: Some completion API might have a last
                                # usage summary response without a token so we
                                # do not want to include as inter-token-latency
                                elif data.get("usage", None) is None:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                most_recent_timestamp = timestamp
                                generated_text += data["choices"][0]["text"]

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

async def async_request_triton_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "completions"
    ), "Triton OpenAI Completions API URL must end with 'completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "temperature": 0.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = remove_prefix(chunk.decode("utf-8"), "data: ")
                        data = json.loads(chunk)

                        if data["choices"][0]["text"]:
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft

                            # Decoding phase
                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # do not want to include as inter-token-latency
                            elif data.get("usage", None) is None:
                                output.itl.append(timestamp -
                                                    most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text += data["choices"][0]["text"]

                    latency = time.perf_counter() - st
                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            #"model": request_func_input.model,
            "model": "model", #"sec-llm-v02",
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 1.0,
            "top_k":50,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            # "ignore_eos": True
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        print("payload is: ", payload)
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        # api_url = "http://inference-internal.yanxi-cap.jd.com/inference/service-wo7xrgru6l/v1/chat/completions"
        #print(api_url)
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                #print(response.status)
                if response.status == 200:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        #print(chunk)
                        chunk = remove_prefix(chunk.decode("utf-8"), "data: ")
                        if chunk == "[DONE]":
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    output.error = response.reason
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output


async def async_request_mindie_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "/chat/completions"
    ), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "model": request_func_input.model.split("/")[-1],
            "messages": [
                {
                    "role": "user",
                    "content": request_func_input.prompt,
                },
            ],
            "temperature": 1,
            "max_tokens": request_func_input.output_len,
            "top_k":50,
            
            "stream": True,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload,
                                    headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk:
                            continue
                        chunk = remove_prefix(chunk.decode("utf-8"), "data: ")
                        if chunk.endswith("[DONE]"):
                            latency = time.perf_counter() - st
                        else:
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            delta = data["choices"][0]["delta"]
                            if delta.get("content", None):
                                # First token
                                if ttft == 0:
                                    ttft = time.perf_counter() - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp -
                                                      most_recent_timestamp)

                                generated_text += delta["content"]

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency
                else:
                    print(output)
                    output.error = response.reason
                    output.success = False
        except Exception as e:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))
    if pbar:
        pbar.update(1)
    return output

async def async_request_generate(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(
        "generate"
    )

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        assert not request_func_input.use_beam_search
        payload = {
            "prompt": request_func_input.prompt,
            "temperature": 1.0,
            "best_of": request_func_input.best_of,
            "max_tokens": request_func_input.output_len,
            "top_k": 50,
            "stream": True,
            "ignore_eos": True,
        }

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        generated_text = ""
        ttft = 0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    async for chunk, _ in response.content.iter_chunks():
                        chunk = chunk.strip()
                        if not chunk:
                            continue

                        chunk = chunk[:-1]
                        data = json.loads(chunk.decode("utf-8"))

                        if data["text"]:
                            timestamp = time.perf_counter()
                            # First token
                            if ttft == 0:
                                ttft = time.perf_counter() - st
                                output.ttft = ttft

                            # Decoding phase
                            # NOTE: Some completion API might have a last
                            # usage summary response without a token so we
                            # do not want to include as inter-token-latency
                            else:
                                output.itl.append(timestamp -
                                                      most_recent_timestamp)

                            most_recent_timestamp = timestamp
                            generated_text = data["text"]

                    latency = time.perf_counter() - st
                    output.generated_text = generated_text
                    output.success = True
                    output.latency = latency

        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output

# Since vllm must support Python 3.8, we can't use str.removeprefix(prefix)
# introduced in Python 3.9
def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "mindie-openai-chat": async_request_mindie_openai_chat_completions,
    "tensorrt-llm": async_request_trt_llm,
    "triton-completions": async_request_triton_completions,
    "mindie-tgi": async_request_npu ,
    "vllm_generate": async_request_generate,
    "pd_generate": async_request_generate,
}
