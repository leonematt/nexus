/* OpenCL runtime library: clGetPlatformIDs()

   Copyright (c) 2011 Kalle Raiskila 
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <assert.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <optional>

#define NXSAPI_LOGGING
#include <nexus-api.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
/* #include <QuartzCore/QuartzCore.hpp> */

#define NXSAPI_LOG_MODULE "metal"

class MetalDevice {
  MTL::Device *device;
  MTL::CommandQueue *queue;
  std::vector<MTL::Buffer *> buffers;
  std::vector<MTL::CommandBuffer *> cmdLists;
  std::vector<MTL::ComputeCommandEncoder *> commands;
  public:
    MetalDevice(MTL::Device *dev) : device(dev) {
      queue = device->newCommandQueue();
    }
    MetalDevice(const MetalDevice &) = delete;

    ~MetalDevice() {
      for (int cid = 0; cid < cmdLists.size(); ++cid)
        releaseCommandBuffer(cid);
      for (int bid = 0; bid < buffers.size(); ++bid)
        releaseBuffer(bid);
      device->release();
    }

    nxs_int createBuffer(size_t size, void *host_data = nullptr) {
      MTL::ResourceOptions bopts = MTL::ResourceStorageModeShared; // unified?
      NXSAPI_LOG(NXSAPI_STATUS_NOTE, "createBuffer " << size);

      MTL::Buffer *buf;
      if (host_data != nullptr)
        buf = device->newBuffer(host_data, size, bopts);
      else
        buf = device->newBuffer(size, bopts);
      buffers.push_back(buf);
      return buffers.size() - 1;
    }

    nxs_status releaseBuffer(nxs_int id) {
      if (id < 0 || id >= buffers.size() || buffers[id] == nullptr)
        return NXS_InvalidBufferSize; // invalid buffer
      NXSAPI_LOG(NXSAPI_STATUS_NOTE, "releaseBuffer " << id);
      buffers[id]->release();
      buffers[id] = nullptr;
      return NXS_Success;
    }

    nxs_int createCommandBuffer() {
      NXSAPI_LOG(NXSAPI_STATUS_NOTE, "createCommandBuffer");
      MTL::CommandBuffer *cmdBuf = queue->commandBuffer();
      MTL::ComputeCommandEncoder *pComputeEncoder =
         cmdBuf->computeCommandEncoder();

      cmdLists.push_back(cmdBuf);
      return cmdLists.size() - 1;
    }

    nxs_status releaseCommandBuffer(nxs_int id) {
      if (id < 0 || id >= cmdLists.size() || cmdLists[id] == nullptr)
        return NXS_InvalidCommandQueue; // invalid buffer
      NXSAPI_LOG(NXSAPI_STATUS_NOTE, "releaseCommandBuffer " << id);
      cmdLists[id]->release();
      cmdLists[id] = nullptr;
      return NXS_Success;
    }

    nxs_int createCommand(nxs_int id = 0) {
      if (id < 0 || id >= cmdLists.size() || commands[id] == nullptr)
        return NXS_InvalidCommandQueue; // invalid buffer OR CREATE ONE???
      NXSAPI_LOG(NXSAPI_STATUS_NOTE, "createCommand");
      MTL::ComputeCommandEncoder *command = cmdLists[id]->computeCommandEncoder();

      commands.push_back(command);
      return commands.size() - 1;
    }

    const MTL::Device *get() const { return device; }
};

class MetalRuntime {
  NS::Array *mDevices;
  std::vector<MetalDevice *> devices;
public:
  MetalRuntime() {
    mDevices = MTL::CopyAllDevices();
    for (int i = 0; i < mDevices->count(); ++i) {
      auto *dev = mDevices->object<MTL::Device>(i);
      devices.push_back(new MetalDevice(dev));
    }
  }
  ~MetalRuntime() {
    for (auto *dev : devices) {
      delete dev;
    }
  }
  const std::vector<MetalDevice *> &getDevices() const {
    return devices;
  }

  std::optional<MetalDevice *> getDevice(nxs_int id) const {
    if (id < 0 || id >= devices.size())
      return std::nullopt;
    return devices[id];
  }
};


const MetalRuntime *getRuntime() {
  static MetalRuntime s_runtime;
  return &s_runtime;
}

/*
 * Get the Runtime properties
 */ 
extern "C" nxs_status NXS_API_CALL
nxsGetRuntimeProperty(
  nxs_uint runtime_property_id,
  void *property_value,
  size_t* property_value_size
)
{
  auto rt = getRuntime();

  /* lookup HIP equivalent */
  /* return value size */
  /* return value */
  switch (runtime_property_id) {
    case NP_Name: {
      const char *name = "metal";
      if (property_value != NULL) {
        strncpy((char*)property_value, name, strlen(name));
      } else if (property_value_size != NULL) {
        *property_value_size = strlen(name);
      }
      break;
    }
    default:
      return NXS_InvalidProperty;
  }
  return NXS_Success;
}

extern "C" nxs_int NXS_API_CALL
nxsGetDeviceCount()
{
  auto &devs = getRuntime()->getDevices();
  return devs.size();
}

/*
 * Get the number of supported platforms on this system. 
 * On POCL, this trivially reduces to 1 - POCL itself.
 */ 
extern "C" nxs_status NXS_API_CALL
nxsGetDeviceProperty(
  nxs_int device_id,
  nxs_uint property_id,
  void *property_value,
  size_t* property_value_size
)
{
  auto &devs = getRuntime()->getDevices();
  if (devs.size() <= device_id)
    return NXS_InvalidDevice;
  auto *device = devs[device_id]->get();

  auto getStr = [&](const char *name, size_t len) {
    if (property_value != NULL) {
      if (property_value_size == NULL)
        return NXS_InvalidArgSize;
      else if (*property_value_size < len)
        return NXS_InvalidArgValue;
      strncpy((char*)property_value, name, len);
    } else if (property_value_size != NULL) {
      *property_value_size = len;
    }
    return NXS_Success;
  };

  switch (property_id) {
    case NP_Name: {
      std::string name = device->name()->cString(NS::StringEncoding::ASCIIStringEncoding);
      return getStr(name.c_str(), name.size()+1);
    }
    case NP_Vendor:
      return getStr("apple", 6);
    case NP_Type:
      return getStr("gpu", 4);
    case NP_Architecture: {
      auto arch = device->architecture();
      std::string name = arch->name()->cString(NS::StringEncoding::ASCIIStringEncoding);
      return getStr(name.c_str(), name.size()+1);
    }

    default:
      return NXS_InvalidProperty;
  }
  return NXS_Success;
}

/*
 * Get the number of supported platforms on this system. 
 * On POCL, this trivially reduces to 1 - POCL itself.
 */ 
extern "C" nxs_status NXS_API_CALL
nxsGetDevicePropertyFromPath(
    nxs_int device_id,
    nxs_uint property_path_count,
    nxs_uint *property_id,
    void *property_value,
    size_t* property_value_size
)
{
  if (property_path_count == 1)
    return nxsGetDeviceProperty(device_id, *property_id, property_value, property_value_size);
  switch (property_id[0]) {
    case NP_CoreSubsystem:
      break;
    case NP_MemorySubsystem:
      break;
    default:
      return NXS_InvalidProperty;
  }
  return NXS_Success;
}

/*
 * Allocate a buffer on the device.
 */ 
extern "C" nxs_int NXS_API_CALL
nxsCreateBuffer(
  nxs_int device_id,
  size_t size,
  nxs_mem_flags flags,
  void* host_ptr
)
{
  auto dev = getRuntime()->getDevice(device_id);
  if (!dev)
    return NXS_InvalidDevice;

  return (*dev)->createBuffer(size, host_ptr);
}


/*
 * Release a buffer on the device.
 */ 
extern "C" nxs_status NXS_API_CALL
nxsReleaseBuffer(
  nxs_int device_id,
  nxs_int buffer_id
)
{
  auto dev = getRuntime()->getDevice(device_id);
  if (!dev)
    return NXS_InvalidDevice;

  return (*dev)->releaseBuffer(buffer_id);
}


/*
 * Allocate a buffer on the device.
 */ 
extern "C" nxs_int NXS_API_CALL
nxsCreateCommandList(
  nxs_int device_id,
  nxs_command_queue_properties properties
)
{
  auto dev = getRuntime()->getDevice(device_id);
  if (!dev)
    return NXS_InvalidDevice;
  return (*dev)->createCommandBuffer();
}

/*
 * Allocate a buffer on the device.
 */ 
extern "C" nxs_status NXS_API_CALL
nxsReleaseCommandList(
  nxs_int device_id,
  nxs_int command_list_id
)
{
  auto dev = getRuntime()->getDevice(device_id);
  if (!dev)
    return NXS_InvalidDevice;
  return (*dev)->releaseCommandBuffer(command_list_id);
}
