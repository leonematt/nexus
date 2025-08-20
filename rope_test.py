import numpy as np
import nexus
import math
import sys

# Helper: build cos/sin cache like vLLM
def create_cos_sin_cache(max_position, rot_dim):
    cos_sin_cache = np.zeros((max_position, rot_dim), dtype=np.float32)
    for pos in range(max_position):
        for dim in range(rot_dim // 2):
            angle = pos / math.pow(10000.0, 2.0 * dim / rot_dim)
            cos_sin_cache[pos, dim] = math.cos(angle)                     # cos
            cos_sin_cache[pos, rot_dim // 2 + dim] = math.sin(angle)      # sin
    return cos_sin_cache.flatten()

def test_rotary_embedding_kernel(runtime_name, kernel_file, kernel_name):
    sys_ = nexus.getSystem()
    runtime = sys_.getRuntime(runtime_name)
    if not runtime:
        print("No runtimes found")
        return 1

    devices = runtime.getDevices()
    if not devices:
        print("No devices found")
        return 1

    dev0 = runtime.getDevice(0)

    # Parameters
    num_tokens = 4
    head_size = 64
    num_heads = 8
    num_kv_heads = 8
    max_position = 100
    rot_dim = head_size
    positions = np.arange(num_tokens, dtype=np.int64)

    # Input tensors
    query = np.random.uniform(-1, 1, size=(num_tokens * num_heads * head_size)).astype(np.float32)
    key   = np.random.uniform(-1, 1, size=(num_tokens * num_kv_heads * head_size)).astype(np.float32)

    cos_sin_cache = create_cos_sin_cache(max_position, rot_dim).astype(np.float32)

    # Buffers
    buf_positions = dev0.createBuffer(positions.nbytes, positions)
    buf_query = dev0.createBuffer(query.nbytes, query)
    buf_key = dev0.createBuffer(key.nbytes, key)
    buf_cache = dev0.createBuffer(cos_sin_cache.nbytes, cos_sin_cache)

    # Kernel
    nlib = dev0.createLibrary(kernel_file)
    kern = nlib.getKernel(kernel_name)
    if not kern:
        print(f"Failed to get kernel: {kernel_name}")
        return 1

    stream0 = dev0.createStream()
    sched = dev0.createSchedule()
    cmd = sched.createCommand(kern)

    # Kernel args (like vLLM signature)
    cmd.setArgument(0, buf_positions)        # positions
    cmd.setArgument(1, buf_query)            # query
    cmd.setArgument(2, buf_key)              # key
    cmd.setArgument(3, buf_cache)            # cos/sin cache
    cmd.setArgument(4, rot_dim)              # rot_dim

    query_stride = num_heads * head_size
    key_stride   = num_kv_heads * head_size
    head_stride  = head_size

    cmd.setArgument(5, query_stride)
    cmd.setArgument(6, key_stride)
    cmd.setArgument(7, head_stride)
    cmd.setArgument(8, num_heads)
    cmd.setArgument(9, num_kv_heads)
    cmd.setArgument(10, head_size)

    # vLLM kernel uses grid=(num_tokens, num_heads, 1), block=(rot_dim/2, 1, 1)
    grid_size = (num_tokens, num_heads, 1)
    block_size = (min(rot_dim // 2, 512), 1, 1)

    cmd.finalize(grid_size, block_size)

    # Run
    sched.run(stream0)

    # Copy back
    result_query = np.zeros_like(query)
    result_key   = np.zeros_like(key)
    buf_query.copy(result_query)
    buf_key.copy(result_key)

    print("Rotary embedding kernel completed successfully!")
    print("First few query values after rotation:", result_query[:5])

    # Validation
    if np.allclose(query, result_query, atol=1e-6):
        print("FAIL: Query values did not change")
        return 1
    else:
        print("PASS: Query values changed")
        return 0

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python rope_test.py <runtime_name> <kernel_file> <kernel_name>")
        sys.exit(1)
    sys.exit(test_rotary_embedding_kernel(sys.argv[1], sys.argv[2], sys.argv[3]))
