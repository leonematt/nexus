#!/usr/bin/env python3

import torch
import triton
import triton.language as tl

@triton.jit
def add_vectors(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
  pid = tl.program_id(0)
  offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
  mask = offsets < n
  x = tl.load(x_ptr + offsets, mask=mask)
  y = tl.load(y_ptr + offsets, mask=mask)
  tl.store(out_ptr + offsets, x + y, mask=mask)

variants = [
  {'BLOCK_SIZE': 256},
  {'BLOCK_SIZE': 512},
  {'BLOCK_SIZE': 1024}
]

def run():
  n = 4096
  
  for variant in variants:
    block_size = variant['BLOCK_SIZE']
    print(f"Running with BLOCK_SIZE={block_size}")

    x = torch.randn(n, device='cuda', dtype=torch.float32)
    y = torch.randn(n, device='cuda', dtype=torch.float32)
    out = torch.empty_like(x)

    grid = (triton.cdiv(n, block_size),)
    add_vectors[grid](x, y, out, n, BLOCK_SIZE=block_size)

    expected = x + y
    max_error = torch.max(torch.abs(out - expected)).item()
    print(f"  Max error: {max_error:.2e}")
    print(f"  Success: {max_error < 1e-5}")

if __name__ == "__main__":
  run()