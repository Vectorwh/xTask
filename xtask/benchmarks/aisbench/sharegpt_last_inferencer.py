"""AISBench compatibility for ShareGPT infer_mode=last."""

import asyncio
import copy
import uuid

from ais_bench.benchmark.models.output import RequestOutput
from ais_bench.benchmark.openicl.icl_inferencer import MultiTurnGenInferencer
from ais_bench.benchmark.utils.prompt import PromptList, get_round_index


class XTaskShareGPTLastInferencer(MultiTurnGenInferencer):
    """Run AISBench last-turn inference without PromptList reflected-add recursion."""

    async def infer_last(self, data, session):
        data = copy.deepcopy(data)
        index = data.pop("index")
        chat = data.pop("prompt")
        data_abbr = data.pop("data_abbr")
        max_out_len = data.pop("max_out_len")
        gold = data.pop("gold", None)
        uid = str(uuid.uuid4()).replace("-", "")

        left_idx, right_idx = get_round_index(chat)
        left_prompt, chat, right_prompt = (
            chat[:left_idx],
            chat[left_idx:right_idx],
            chat[right_idx:],
        )
        bot_indices = [
            bot_index
            for bot_index, item in enumerate(chat)
            if item["role"] == "BOT"
        ]
        history = PromptList(chat[: bot_indices[-1]])

        # AISBench's PromptList.__radd__ recurses for list + PromptList. Build
        # the combined sequence explicitly while preserving the prompt type.
        prompt = PromptList([*left_prompt, *history, *right_prompt])
        history = await asyncio.to_thread(
            self.model.parse_template, prompt, mode="gen"
        )

        output = RequestOutput(self.perf_mode)
        output.turn_id = 0
        output.uuid = uid
        max_out_len = (
            max_out_len[-1] if isinstance(max_out_len, list) else max_out_len
        )
        await self.status_counter.post()
        await self.model.generate(history, max_out_len, output, session=session, **data)
        if output.success:
            await self.status_counter.rev()
        else:
            await self.status_counter.failed()
        await self.status_counter.finish()
        await self.output_handler.report_cache_info(
            index, history, output, data_abbr, gold
        )
