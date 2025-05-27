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

#include <nexus-api.h>

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define MTK_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
/* #include <QuartzCore/QuartzCore.hpp> */

class MetalDevice {
  MTL::Device *device;
  std::vector<MTL::Buffer *> buffers;
  public:
    MetalDevice(MTL::Device *dev) : device(dev) {}
    MetalDevice(const MetalDevice &) = delete;
    ~MetalDevice() {}

    nxs_uint allocateBuffer(size_t size, void *host_data = nullptr) {
      MTL::ResourceOptions bopts = MTL::ResourceCPUCacheModeDefaultCache;
      MTL::Buffer *buf;
      if (host_data != nullptr)
        buf = device->newBuffer(host_data, size, bopts);
      else
        buf = device->newBuffer(size, bopts);
      buffers.push_back(buf);
      return buffers.size() - 1;
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
};


const MetalRuntime *getRuntime() {
  static MetalRuntime s_runtime;
  return &s_runtime;
}

/*
 * Get the Runtime properties
 */ 
extern "C" nxs_int NXS_API_CALL
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
      break;
  }
  return NXS_SUCCESS;
}

extern "C" nxs_int NXS_API_CALL
nxsGetDeviceCount(
  nxs_uint* num_devices
)
{
  auto &devs = getRuntime()->getDevices();
  *num_devices = devs.size();
  return NXS_SUCCESS;
}

/*
 * Get the number of supported platforms on this system. 
 * On POCL, this trivially reduces to 1 - POCL itself.
 */ 
extern "C" nxs_int NXS_API_CALL
nxsGetDeviceProperty(
  nxs_uint device_id,
  nxs_uint property_id,
  void *property_value,
  size_t* property_value_size
)
{
  auto &devs = getRuntime()->getDevices();
  if (devs.size() <= device_id)
    return -1; // ERR
  auto *device = devs[device_id]->get();


  auto getStr = [&](const char *name, size_t len) {
    if (property_value != NULL) {
      if (property_value_size == NULL)
        return NXS_INVALID_ARG_SIZE;
      else if (*property_value_size < len)
        return NXS_INVALID_ARG_SIZE;
      strncpy((char*)property_value, name, len);
    } else if (property_value_size != NULL) {
      *property_value_size = len;
    }
    return NXS_SUCCESS;
  };

  switch (property_id) {
    case NP_Name: {
      std::string name = device->name()->cString(NS::StringEncoding::ASCIIStringEncoding);
      return getStr(name.c_str(), name.size());
    }
    case NP_Vendor:
      return getStr("apple", 6);
    case NP_Type:
      return getStr("gpu", 4);
    case NP_Architecture: {
      auto arch = device->architecture();
      std::string name = arch->name()->cString(NS::StringEncoding::ASCIIStringEncoding);
      return getStr(name.c_str(), name.size());
    }

    default:
      return NXS_INVALID_PROPERTY;
  }
  return NXS_SUCCESS;
}

/*
 * Get the number of supported platforms on this system. 
 * On POCL, this trivially reduces to 1 - POCL itself.
 */ 
extern "C" nxs_int NXS_API_CALL
nxsGetDevicePropertyFromPath(
    nxs_uint device_id,
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
      return -1; // ERR
  }
  return NXS_SUCCESS;
}

#define NXS_SET_RETURN_STATUS(STATUS) { if (errcode_ret != NULL) *errcode_ret = STATUS; }

/*
 * Allocate a buffer on the device.
 */ 
extern "C" nxs_uint NXS_API_CALL
nxsCreateBuffer(
  nxs_uint device_id,
  size_t size,
  nxs_mem_flags flags,
  void* host_ptr,
  nxs_int* errcode_ret
)
{
  NXS_SET_RETURN_STATUS(NXS_SUCCESS);
  auto &devs = getRuntime()->getDevices();
  if (devs.size() <= device_id) {
    NXS_SET_RETURN_STATUS(NXS_INVALID_DEVICE);
    return -1; // Invalid device_id
  }
  // CHECK valid device_id
  auto *device = devs[device_id];

  nxs_uint bid = device->allocateBuffer(size, host_ptr);
  return bid;
}