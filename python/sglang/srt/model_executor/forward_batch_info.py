"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""ModelRunner runs the forward passes of the models."""
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import BaseTokenToKVPool, ReqToTokenPool

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class ForwardMode(IntEnum):
    # Prefill a new sequence. This is deprecated now. "EXTEND" covers this case.
    PREFILL = auto()
    # Extend a sequence. The KV cache of the first part of the sequence is already computed (e.g., system prompt).
    EXTEND = auto()
    # Decode one token.
    DECODE = auto()


@dataclass
class InputMetadata:
    """Store all inforamtion of a forward pass."""

    forward_mode: ForwardMode
    batch_size: int
    req_pool_indices: torch.Tensor
    seq_lens: torch.Tensor
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: BaseTokenToKVPool

    # Output location of the KV cache
    out_cache_loc: torch.Tensor

    total_num_tokens: int = None

    # Position information
    positions: torch.Tensor = None

    # For extend
    extend_seq_lens: torch.Tensor = None
    extend_start_loc: torch.Tensor = None
    extend_no_prefix: bool = None

    # For logprob
    return_logprob: bool = False
    top_logprobs_nums: List[int] = None
    extend_seq_lens_cpu: List[int] = None
    logprob_start_lens_cpu: List[int] = None

    # For multimodal
    pixel_values: List[torch.Tensor] = None
    image_sizes: List[List[int]] = None
    image_offsets: List[int] = None

    # Trition attention backend
    triton_max_seq_len: int = 0
    triton_max_extend_len: int = 0
    triton_start_loc: torch.Tensor = None
    triton_prefix_lens: torch.Tensor = None

    # FlashInfer attention backend
    flashinfer_prefill_wrapper_ragged: "BatchPrefillWithRaggedKVCacheWrapper" = None
    flashinfer_prefill_wrapper_paged: "BatchPrefillWithPagedKVCacheWrapper" = None
    flashinfer_decode_wrapper: "BatchDecodeWithPagedKVCacheWrapper" = None
    flashinfer_use_ragged: bool = False

    def init_multimuldal_info(self, batch: ScheduleBatch):
        reqs = batch.reqs
        self.pixel_values = [r.pixel_values for r in reqs]
        self.image_sizes = [r.image_size for r in reqs]
        self.image_offsets = [
            (
                (r.image_offset - batch.prefix_lens_cpu[i])
                if r.image_offset is not None
                else 0
            )
            for i, r in enumerate(reqs)
        ]

    def compute_positions(self, batch: ScheduleBatch):
        position_ids_offsets = batch.position_ids_offsets

        if self.forward_mode == ForwardMode.DECODE:
            if True:
                self.positions = self.seq_lens - 1
                print(f"1 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_positions, self.positions.shape: {self.positions.shape}")
            else:
                # Deprecated
                self.positions = (self.seq_lens - 1) + position_ids_offsets
        else:
            if True:
                self.positions = torch.tensor(
                    np.concatenate(
                        [
                            np.arange(batch.prefix_lens_cpu[i], len(req.fill_ids))
                            for i, req in enumerate(batch.reqs)
                        ],
                        axis=0,
                    ),
                    device="cuda",
                )
                for i, req in enumerate(batch.reqs):
                    print(f"1.5 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_positions, req i:{i} and batch.prefix_lens_cpu[i]:{batch.prefix_lens_cpu[i]} and len(req.fill_ids):{len(req.fill_ids)}")
                print(f"2 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_positions, self.positions.shape: {self.positions.shape}")
            else:
                # Deprecated
                position_ids_offsets_cpu = position_ids_offsets.cpu().numpy()
                self.positions = torch.tensor(
                    np.concatenate(
                        [
                            np.arange(
                                batch.prefix_lens_cpu[i] + position_ids_offsets_cpu[i],
                                len(req.fill_ids) + position_ids_offsets_cpu[i],
                            )
                            for i, req in enumerate(batch.reqs)
                        ],
                        axis=0,
                    ),
                    device="cuda",
                )

        # Positions should be in long type
        self.positions = self.positions.to(torch.int64)

    #xiao 0827 这个很重要
    def compute_extend_infos(self, batch: ScheduleBatch):
        print(f"1 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, self.forward_mode: {self.forward_mode}")
        if self.forward_mode == ForwardMode.DECODE:
            #xiao: 0827 为什么decode的时候不需要extend_seq_lens等信息
            print(f"2 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, self.forward_mode decode")
            self.extend_seq_lens = self.extend_start_loc = self.extend_no_prefix = None
            self.extend_seq_lens_cpu = self.logprob_start_lens_cpu = None
        else:
            #xiao: 0902 这些设置的意义是什么
            extend_lens_cpu = [
                len(r.fill_ids) - batch.prefix_lens_cpu[i]
                for i, r in enumerate(batch.reqs)
            ]
            for i, r in enumerate(batch.reqs):
                print(f"2.5 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, i:{i} and r.fill_ids:{len(r.fill_ids)} and batch.prefix_lens_cpu[i]:{batch.prefix_lens_cpu[i]}")
            print(f"3 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, extend_lens_cpu: {extend_lens_cpu}")
            self.extend_seq_lens = torch.tensor(extend_lens_cpu, device="cuda")
            self.extend_start_loc = torch.zeros_like(self.seq_lens)
            self.extend_start_loc[1:] = torch.cumsum(self.extend_seq_lens[:-1], dim=0)
            self.extend_no_prefix = all(l == 0 for l in batch.prefix_lens_cpu)
            print(f"4 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, self.extend_seq_lens: {self.extend_seq_lens.shape}")
            print(f"5 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, self.extend_start_loc: {self.extend_start_loc.shape}")
            print(f"6 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, self.extend_no_prefix: {self.extend_no_prefix}")

            self.extend_seq_lens_cpu = extend_lens_cpu
            self.logprob_start_lens_cpu = [
                (
                    min(
                        req.logprob_start_len - batch.prefix_lens_cpu[i],
                        extend_lens_cpu[i] - 1,
                    )
                    if req.logprob_start_len >= batch.prefix_lens_cpu[i]
                    else extend_lens_cpu[i] - 1  # Fake extend, actually decode
                )
                for i, req in enumerate(batch.reqs)
            ]
            for i, req in enumerate(batch.reqs):
                if req.logprob_start_len < batch.prefix_lens_cpu[i]:
                    x =      min(
                        req.logprob_start_len - batch.prefix_lens_cpu[i],
                        extend_lens_cpu[i] - 1,
                    )
                    print(f"6.3 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, i:{i} and x:{x}")
                else:
                    x = extend_lens_cpu[i] - 1
                    print(f"6.4 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, i:{i} and x:{x}")
                #print(f"6.5 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, i:{i} and x:{x}")
            print(f"7 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::compute_extend_infos, self.extend_seq_lens_cpu: {self.extend_seq_lens_cpu}")

    @classmethod
    def from_schedule_batch(
        cls,
        model_runner: "ModelRunner",
        batch: ScheduleBatch,
        forward_mode: ForwardMode,
    ):
        ret = cls(
            forward_mode=forward_mode,
            batch_size=batch.batch_size(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool=model_runner.token_to_kv_pool,
            out_cache_loc=batch.out_cache_loc,
            return_logprob=batch.return_logprob,
            top_logprobs_nums=batch.top_logprobs_nums,
        ) #0xiao 初始化InputMetadata类

        ret.compute_positions(batch)

        ret.compute_extend_infos(batch)
        print(f"0 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::from_schedule_batch, forward_mode != ForwardMode.DECODE:{forward_mode != ForwardMode.DECODE} and model_runner.server_args.disable_flashinfer:{model_runner.server_args.disable_flashinfer}")
        if (
            forward_mode != ForwardMode.DECODE
            or model_runner.server_args.disable_flashinfer
        ):
            ret.total_num_tokens = int(torch.sum(ret.seq_lens))
            print(f"1 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::from_schedule_batch, ret.total_num_tokens: {ret.total_num_tokens}")

        if forward_mode != ForwardMode.DECODE:
            print(f"2 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::from_schedule_batch, forward_mode != ForwardMode.DECODE")
            ret.init_multimuldal_info(batch)

        if model_runner.server_args.disable_flashinfer:
            ret.init_triton_args(batch) #xiao 0902 这个函数是干什么的 0919：这个函数是初始化triton的参数

        flashinfer_use_ragged = False
        #xiao:0902 设置flashinfer 的用法
        if not model_runner.server_args.disable_flashinfer:
            print(f"2.5 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::from_schedule_batch, int(torch.sum(ret.seq_lens)) :{int(torch.sum(ret.seq_lens))}")
            if (
                forward_mode != ForwardMode.DECODE
                and int(torch.sum(ret.seq_lens)) > 4096
                and model_runner.sliding_window_size is None
            ): #TODO xiao: 0919 在哪里更新ret.seq_lens?
                flashinfer_use_ragged = True
                print(f"3 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::from_schedule_batch, flashinfer_use_ragged: {flashinfer_use_ragged}")
            print(f"4 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::from_schedule_batch, init_flashinfer_handlers")
            ret.init_flashinfer_handlers(
                model_runner, batch.prefix_lens_cpu, flashinfer_use_ragged
            )

        return ret

    def init_triton_args(self, batch: ScheduleBatch):
        """Init auxiliary variables for triton attention backend."""
        self.triton_max_seq_len = int(torch.max(self.seq_lens))
        self.triton_start_loc = torch.zeros_like(self.seq_lens, dtype=torch.int32)
        self.triton_start_loc[1:] = torch.cumsum(self.seq_lens[:-1], dim=0)

        if self.forward_mode == ForwardMode.DECODE:
            self.triton_max_extend_len = None
        else:
            self.triton_prefix_lens = torch.tensor(batch.prefix_lens_cpu, device="cuda")
            extend_seq_lens = self.seq_lens - self.triton_prefix_lens
            self.triton_max_extend_len = int(torch.max(extend_seq_lens))

    def init_flashinfer_handlers(
        self,
        model_runner,
        prefix_lens_cpu,
        flashinfer_use_ragged,
    ):
        if self.forward_mode != ForwardMode.DECODE:
            
            prefix_lens = torch.tensor(prefix_lens_cpu, device="cuda")
            print(f"1 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::init_flashinfer_handlers, Extend, prefix_lens.shape: {prefix_lens.shape}")  
        else:
            prefix_lens = None
            print(f"2 python/sglang/srt/model_executor/forward_batch_info.py InputMetadata::init_flashinfer_handlers, Decode prefix_lens: {prefix_lens}")

        update_flashinfer_indices(
            self.forward_mode,
            model_runner,
            self.req_pool_indices,
            self.seq_lens,
            prefix_lens,
            flashinfer_use_ragged=flashinfer_use_ragged,
        )

        (
            self.flashinfer_prefill_wrapper_ragged,
            self.flashinfer_prefill_wrapper_paged,
            self.flashinfer_decode_wrapper,
            self.flashinfer_use_ragged,
        ) = (
            model_runner.flashinfer_prefill_wrapper_ragged,
            model_runner.flashinfer_prefill_wrapper_paged,
            model_runner.flashinfer_decode_wrapper,
            flashinfer_use_ragged,
        )


def update_flashinfer_indices(
    forward_mode,
    model_runner,
    req_pool_indices,
    seq_lens,
    prefix_lens,
    flashinfer_decode_wrapper=None,
    flashinfer_use_ragged=False,
):
    """Init auxiliary variables for FlashInfer attention backend."""
    num_qo_heads = model_runner.model_config.num_attention_heads // model_runner.tp_size
    num_kv_heads = model_runner.model_config.get_num_kv_heads(model_runner.tp_size)
    head_dim = model_runner.model_config.head_dim
    batch_size = len(req_pool_indices)
    if forward_mode != ForwardMode.DECODE:
        print(f"1 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, forward_mode != ForwardMode.DECODE")
    else:
        print(f"2 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, forward_mode == ForwardMode.DECODE")
    print(f"3 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, num_qo_heads: {num_qo_heads}")
    print(f"4 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, num_kv_heads: {num_kv_heads}")
    print(f"5 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, head_dim: {head_dim}")
    print(f"6 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, batch_size: {batch_size}")
    print(f"7 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, flashinfer_use_ragged: {flashinfer_use_ragged}")
    print(f"8 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, model_runner.sliding_window_size is None : {model_runner.sliding_window_size is None}")
    if model_runner.sliding_window_size is None:
        if flashinfer_use_ragged:
            paged_kernel_lens = prefix_lens
            print(f"9 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, flashinfer_use_ragged true, paged_kernel_lens = prefix_lens  paged_kernel_lens: {paged_kernel_lens}")
        else:
            paged_kernel_lens = seq_lens
            print(f"10 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, flashinfer_use_ragged false,paged_kernel_lens = seq_lens  paged_kernel_lens: {paged_kernel_lens}")

        kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
        kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)#计算 input 张量在指定维度 dim 上的累积和。
        req_pool_indices_cpu = req_pool_indices.cpu().numpy()
        print(f"11 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, req_pool_indices_cpu: {req_pool_indices_cpu.shape} and req_pool_indices_cpu: {req_pool_indices_cpu}")
        paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
        print(f"12 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, paged_kernel_lens_cpu: {paged_kernel_lens_cpu.shape} and paged_kernel_lens_cpu: {paged_kernel_lens_cpu}")
        kv_indices = torch.cat(
            [
                model_runner.req_to_token_pool.req_to_token[
                    req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]
                ]
                for i in range(batch_size)
            ],
            dim=0,
        ).contiguous()
        for i in range(batch_size):
            x =  model_runner.req_to_token_pool.req_to_token[
                    req_pool_indices_cpu[i], : paged_kernel_lens_cpu[i]
                ]
            print(f"13 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, i:{i} and x.shape: {x.shape}")
        kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")

        if forward_mode == ForwardMode.DECODE:
            # CUDA graph uses different flashinfer_decode_wrapper
            print(f"13.5 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, forward_mode == ForwardMode.DECODE")
            if flashinfer_decode_wrapper is None:
                flashinfer_decode_wrapper = model_runner.flashinfer_decode_wrapper
            print(f"13.6 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, kv_indptr.shape: {kv_indptr.shape}")
            print(f"13.7 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, kv_indices.shape: {kv_indices.shape}")
            print(f"13.8 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, kv_last_page_len.shape: {kv_last_page_len.shape}")
            print("14 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, flashinfer_decode_wrapper:{flashinfer_decode_wrapper}")
            flashinfer_decode_wrapper.end_forward()
            flashinfer_decode_wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
            )
        else:
            # extend part
            print(f"15 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, extend part")
            qo_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
            qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

            if flashinfer_use_ragged:
                print(f"16 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, model_runner.flashinfer_prefill_wrapper_ragged:{model_runner.flashinfer_prefill_wrapper_ragged}")
                model_runner.flashinfer_prefill_wrapper_ragged.end_forward()
                model_runner.flashinfer_prefill_wrapper_ragged.begin_forward(
                    qo_indptr,
                    qo_indptr,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                )

            # cached part
            print(f"17 python/sglang/srt/model_executor/forward_batch_info.py update_flashinfer_indices, model_runner.flashinfer_prefill_wrapper_paged:{model_runner.flashinfer_prefill_wrapper_paged}")
            model_runner.flashinfer_prefill_wrapper_paged.end_forward()
            model_runner.flashinfer_prefill_wrapper_paged.begin_forward(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,   
                head_dim,
                1,
            )
    else:
        # window attention use paged only
        kv_last_page_len = torch.ones((batch_size,), dtype=torch.int32, device="cuda")
        for wrapper_id in range(2):
            if wrapper_id == 0:
                if forward_mode == ForwardMode.DECODE:
                    paged_kernel_lens = torch.minimum(
                        seq_lens, torch.tensor(model_runner.sliding_window_size + 1)
                    )
                else:
                    paged_kernel_lens = torch.minimum(
                        seq_lens,
                        torch.tensor(model_runner.sliding_window_size)
                        + seq_lens
                        - prefix_lens,
                    )
            else:
                paged_kernel_lens = seq_lens

            kv_start_idx = seq_lens - paged_kernel_lens

            kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int32, device="cuda")
            kv_indptr[1:] = torch.cumsum(paged_kernel_lens, dim=0)
            req_pool_indices_cpu = req_pool_indices.cpu().numpy()
            paged_kernel_lens_cpu = paged_kernel_lens.cpu().numpy()
            kv_indices = torch.cat(
                [
                    model_runner.req_to_token_pool.req_to_token[
                        req_pool_indices_cpu[i],
                        kv_start_idx[i] : kv_start_idx[i] + paged_kernel_lens_cpu[i],
                    ]
                    for i in range(batch_size)
                ],
                dim=0,
            ).contiguous()

            if forward_mode == ForwardMode.DECODE:
                # CUDA graph uses different flashinfer_decode_wrapper
                if flashinfer_decode_wrapper is None:
                    flashinfer_decode_wrapper = model_runner.flashinfer_decode_wrapper

                flashinfer_decode_wrapper[wrapper_id].end_forward()
                flashinfer_decode_wrapper[wrapper_id].begin_forward(
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                )
            else:
                # extend part
                qo_indptr = torch.zeros(
                    (batch_size + 1,), dtype=torch.int32, device="cuda"
                )
                qo_indptr[1:] = torch.cumsum(seq_lens - prefix_lens, dim=0)

                model_runner.flashinfer_prefill_wrapper_paged[wrapper_id].end_forward()
                model_runner.flashinfer_prefill_wrapper_paged[wrapper_id].begin_forward(
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_last_page_len,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    1,
                )
