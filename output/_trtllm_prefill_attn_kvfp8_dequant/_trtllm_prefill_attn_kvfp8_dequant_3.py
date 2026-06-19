from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import torch

import logging
import triton
import triton.language as tl

PAD_SLOT_ID = -1

logger = logging.getLogger(__name__)

FLASHINFER_WORKSPACE_BUFFER_SIZE_BATCH_INVARIANT = 2048 * 1024 * 1024

FP8_DTYPE = torch.float8_e4m3fn  # NPU FP8 dtype
FP4_DTYPE = torch.uint8

trtllm_gen_workspace_buffer = None

@triton.jit
def _trtllm_prefill_attn_kvfp8_dequant(
    kv_cache_ptr,
    block_tables_prefill_ptr,
    block_table_stride,
    mock_kv_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    K_CACHE_STRIDE: tl.constexpr,
    KV_CACHE_STRIDE: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    mock_block_table_idx = tl.program_id(1).to(tl.int64)
    
    # Double buffering: prefetch next block info when valid
    next_orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx + 1,
        mask=(mock_block_table_idx + 1 < block_table_stride),
        other=0
    ).to(tl.int64)
    
    orig_page_num = tl.load(
        block_tables_prefill_ptr + batch_idx * block_table_stride + mock_block_table_idx
    ).to(tl.int64)
    if orig_page_num <= 0:
        return
        
    dequant_dtype = mock_kv_cache_ptr.dtype.element_ty
    k_scale_val = tl.load(k_scale_ptr)
    v_scale_val = tl.load(v_scale_ptr)

    # Streamlined K cache processing with contiguous access
    k_offset_curr = orig_page_num * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    fp8_k_vals_curr = tl.load(kv_cache_ptr + k_offset_curr)
    dequantized_k_vals = fp8_k_vals_curr.to(tl.float32) * k_scale_val
    
    mock_cache_offset = (
        batch_idx * block_table_stride + mock_block_table_idx + 1
    ) * KV_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_k_vals.to(dequant_dtype))

    # Streamlined V cache processing with contiguous access
    v_offset_curr = orig_page_num * KV_CACHE_STRIDE + K_CACHE_STRIDE + tl.arange(0, K_CACHE_STRIDE)
    fp8_v_vals_curr = tl.load(kv_cache_ptr + v_offset_curr)
    dequantized_v_vals = fp8_v_vals_curr.to(tl.float32) * v_scale_val
    
    mock_cache_offset = (
        (batch_idx * block_table_stride + mock_block_table_idx + 1) * KV_CACHE_STRIDE
        + K_CACHE_STRIDE
        + tl.arange(0, K_CACHE_STRIDE)
    )
    tl.store(mock_kv_cache_ptr + mock_cache_offset, dequantized_v_vals.to(dequant_dtype))

def trtllm_prefill_attn_kvfp8_dequant(
    kv_cache: torch.Tensor,
    block_tables_prefill: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    dequant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert kv_cache.device.type == 'npu', "kv_cache must be on NPU"
    assert block_tables_prefill.device.type == 'npu', "block_tables_prefill must be on NPU"
    assert k_scale.device.type == 'npu', "k_scale must be on NPU"
    assert v_scale.device.type == 'npu', "v_scale must be on NPU"
    
    batch_size, num_of_page_per_token = block_tables_prefill.shape
    s = kv_cache.shape
    assert s[1] == 2
    assert dequant_dtype in (torch.bfloat16, torch.float16)
    k_cache_stride = s[2] * s[3] * s[4]
    kv_cache_stride = k_cache_stride * s[1]
    new_s = (batch_size * num_of_page_per_token + 1, s[1], s[2], s[3], s[4])
    mock_kv_cache = torch.empty(new_s, dtype=dequant_dtype, device='npu')
    mock_block_table = torch.arange(
        start=1,
        end=batch_size * num_of_page_per_token + 1,
        dtype=torch.int32,
        device='npu',
    ).reshape(batch_size, num_of_page_per_token)
    grid = (batch_size, num_of_page_per_token)
    _trtllm_prefill_attn_kvfp8_dequant[grid](
        kv_cache,
        block_tables_prefill,
        num_of_page_per_token,
        mock_kv_cache,
        k_scale,
        v_scale,
        k_cache_stride,
        kv_cache_stride,
    )
    return mock_kv_cache, mock_block_table