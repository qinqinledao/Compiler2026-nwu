import torch
import triton
import triton.language as tl

@triton.jit
def _count_expert_num_tokens(
    topk_ids_ptr,
    expert_num_tokens_ptr,
    num_experts: tl.constexpr,
    topk_numel: tl.constexpr,
    expert_map_ptr,
    HAS_EXPERT_MAP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    curr_expert = tl.program_id(0)
    
    if curr_expert >= num_experts:
        return
    
    offsets = tl.arange(0, BLOCK_SIZE)
    num_blocks = tl.cdiv(topk_numel, BLOCK_SIZE)
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    base_ptr = topk_ids_ptr + offsets
    
    next_block_start = 0
    next_mask = offsets < (topk_numel - next_block_start)
    next_expert_ids = tl.load(base_ptr + next_block_start, mask=next_mask, other=-1)
    
    for x in range(num_blocks):
        curr_block_start = next_block_start
        curr_mask = next_mask
        curr_expert_ids = next_expert_ids
        
        next_block_start = (x + 1) * BLOCK_SIZE
        next_mask = offsets < (topk_numel - next_block_start)
        if x + 1 < num_blocks:
            next_expert_ids = tl.load(base_ptr + next_block_start, mask=next_mask, other=-1)
        
        if HAS_EXPERT_MAP:
            map_mask = curr_expert_ids >= 0
            curr_expert_ids = tl.load(expert_map_ptr + curr_expert_ids, mask=map_mask, other=-1)
        
        acc += tl.where(curr_expert_ids == curr_expert, 1, 0)
    
    tl.store(expert_num_tokens_ptr + curr_expert, tl.sum(acc))

def count_expert_num_tokens(
    topk_ids: torch.Tensor,
    num_local_experts: int,
    expert_map: torch.Tensor | None
) -> torch.Tensor:
    assert topk_ids.dtype.is_signed, "Kernel requires signed dtype for masking"
    expert_num_tokens = torch.empty(
        (num_local_experts), device=topk_ids.device, dtype=torch.int32
    )
    
    grid = num_local_experts
    topk_numel = topk_ids.numel()
    
    BLOCK_SIZE = 128
    if topk_numel < 512:
        BLOCK_SIZE = 64
    if topk_numel < 256:
        BLOCK_SIZE = 32
    if topk_numel < 128:
        BLOCK_SIZE = 16

    _count_expert_num_tokens[(grid,)](
        topk_ids,
        expert_num_tokens,
        num_local_experts,
        topk_numel,
        expert_map,
        HAS_EXPERT_MAP=expert_map is not None,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return expert_num_tokens