#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>

#include <iostream>

int main(int argc, const char *argv[]) {
  std::cout << std::fixed << std::setprecision(6);

  // --- 1. Setup Device and Command Queue ---
  MTL::Device *pDevice = MTL::CreateSystemDefaultDevice();
  if (!pDevice) {
    std::cerr << "Failed to get Metal device." << std::endl;
    return 1;
  }
  std::cout << "[DEVICE]: " << pDevice->name()->utf8String() << std::endl
            << std::endl;

  MTL::CommandQueue *pCommandQueue = pDevice->newCommandQueue();
  if (!pCommandQueue) {
    std::cerr << "Failed to create command queue." << std::endl;
    pDevice->release();
    return 1;
  }

  // --- 2. Load Library and Get Function ---
  NS::Error *pError = nullptr;
  std::string libraryPathStr = getLibraryPath(
      "add_vectors"); // Name of the .metal file without extension
  MTL::Library *pLibrary = pDevice->newLibrary(
      NS::String::string(libraryPathStr.c_str(), NS::UTF8StringEncoding),
      &pError);

  if (!pLibrary) {
    std::cerr << "Failed to load library: " << libraryPathStr << std::endl;
    if (pError) {
      std::cerr << "Error: " << pError->localizedDescription()->utf8String()
                << std::endl;
      pError->release();
    }
    pCommandQueue->release();
    pDevice->release();
    return 1;
  }

  MTL::Function *pAddFunction = pLibrary->newFunction(
      NS::String::string("add_vectors", NS::UTF8StringEncoding));
  if (!pAddFunction) {
    std::cerr << "Failed to find function 'add_vectors' in library."
              << std::endl;
    pLibrary->release();
    pCommandQueue->release();
    pDevice->release();
    return 1;
  }

  // --- 3. Create Compute Pipeline State ---
  MTL::ComputePipelineState *pAddPSO =
      pDevice->newComputePipelineState(pAddFunction, &pError);
  if (!pAddPSO) {
    std::cerr << "Failed to create compute pipeline state." << std::endl;
    if (pError) {
      std::cerr << "Error: " << pError->localizedDescription()->utf8String()
                << std::endl;
      pError->release();
    }
    pAddFunction->release();
    pLibrary->release();
    pCommandQueue->release();
    pDevice->release();
    return 1;
  }

  // --- 4. Prepare Data and Buffers ---
  std::vector<float> vecA(ARRAY_LENGTH);
  std::vector<float> vecB(ARRAY_LENGTH);
  std::vector<float> vecResult_GPU(ARRAY_LENGTH); // For GPU result
  std::vector<float> vecResult_CPU(ARRAY_LENGTH); // For CPU verification

  for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
    vecA[i] = static_cast<float>(i);
    vecB[i] = static_cast<float>(ARRAY_LENGTH - i);
  }

  // --- 5. CPU Execution and Timing ---
  auto cpu_start_time = std::chrono::high_resolution_clock::now();
  cpu_add_vectors(vecA, vecB, vecResult_CPU, ARRAY_LENGTH);
  auto cpu_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpu_duration_ms =
      cpu_end_time - cpu_start_time;
  std::cout << "[CPU] add_vectors execution time: " << cpu_duration_ms.count()
            << " ms" << std::endl
            << std::endl;

  // --- 6. GPU Execution: Preprare Buffers ---

  size_t bufferSize = ARRAY_LENGTH * sizeof(float);

  MTL::Buffer *pBufferA =
      pDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  MTL::Buffer *pBufferB =
      pDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  MTL::Buffer *pBufferResult =
      pDevice->newBuffer(bufferSize, MTL::ResourceStorageModeShared);

  // Copy data to buffers
  memcpy(pBufferA->contents(), vecA.data(), bufferSize);
  memcpy(pBufferB->contents(), vecB.data(), bufferSize);

  // --- 7. Create Command Buffer, Encode, Commit, Wait ---
  MTL::CommandBuffer *pCommandBuffer = pCommandQueue->commandBuffer();
  MTL::ComputeCommandEncoder *pComputeEncoder =
      pCommandBuffer->computeCommandEncoder();

  // --- 7a. Encode Commands ---
  pComputeEncoder->setComputePipelineState(pAddPSO);
  pComputeEncoder->setBuffer(pBufferA, 0, 0);
  pComputeEncoder->setBuffer(pBufferB, 0, 1);
  pComputeEncoder->setBuffer(pBufferResult, 0, 2);

  MTL::Size gridSize = MTL::Size(ARRAY_LENGTH, 1, 1);
  NS::UInteger threadGroupSize = pAddPSO->maxTotalThreadsPerThreadgroup();
  if (threadGroupSize > ARRAY_LENGTH) {
    threadGroupSize = ARRAY_LENGTH;
  }
  MTL::Size threadgroupSize = MTL::Size(threadGroupSize, 1, 1);

  pComputeEncoder->dispatchThreads(gridSize, threadgroupSize);
  pComputeEncoder->endEncoding();

  // --- 7b. Commit and Wait ---
  pCommandBuffer->commit();
  pCommandBuffer->waitUntilCompleted(); // Synchronous wait for simplicity

  // --- 8. Read Back Results & Benchmark ---
  double gpuStartTime =
      pCommandBuffer->GPUEndTime(); // Using GPU start/end for overall time
  double gpuEndTime =
      pCommandBuffer
          ->kernelEndTime(); // Using kernel start/end for just kernel time

  if (pCommandBuffer->status() == MTL::CommandBufferStatusError) {
    std::cerr << "Command Buffer Error: "
              << pCommandBuffer->error()->localizedDescription()->utf8String()
              << std::endl;
  } else {
    memcpy(vecResult_GPU.data(), pBufferResult->contents(), bufferSize);

    std::cout << "[GPU] Kernel Execution Time: "
              << (pCommandBuffer->kernelEndTime() -
                  pCommandBuffer->kernelStartTime()) *
                     1000.0
              << " ms" << std::endl;
    std::cout << "[GPU] Command Buffer Time: "
              << (pCommandBuffer->GPUEndTime() -
                  pCommandBuffer->GPUStartTime()) *
                     1000.0
              << " ms" << std::endl
              << std::endl;
  }

  // --- 9. Verification ---
  bool all_match = true;
  for (unsigned int i = 0; i < ARRAY_LENGTH; ++i) {
    if (std::abs(vecResult_CPU[i] - vecResult_GPU[i]) >
        1e-5) { // Tolerance for float comparison
      std::cerr << "Mismatch at index " << i << ": CPU=" << vecResult_CPU[i]
                << ", GPU=" << vecResult_GPU[i] << std::endl;
      all_match = false;
      // break; // Uncomment to stop at first error
    }
  }

  if (all_match) {
    std::cout << "Verification successful: GPU and CPU results match."
              << std::endl;
  } else {
    std::cout << "Verification FAILED: GPU and CPU results DIVERGE."
              << std::endl;
  }

  // --- 10. Cleanup ---
  pBufferA->release();
  pBufferB->release();
  pBufferResult->release();
  pAddPSO->release();
  pAddFunction->release();
  pLibrary->release();
  pCommandQueue->release();
  pDevice->release();

  return 0;
}
