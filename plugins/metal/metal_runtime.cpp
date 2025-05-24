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

class MetalRuntime {
  NS::Array *devices;
public:
  MetalRuntime() {
    devices = MTL::CopyAllDevices();
  }
  const NS::Array *getDevices() const {
    assert(devices);
    return devices;
  }
};


const MetalRuntime &getRuntime() {
  static MetalRuntime s_runtime;
  return s_runtime;
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
  nxs_device_type device_type,
  nxs_uint* num_devices
)
{
  if (device_type != NXS_DEVICE_TYPE_GPU)
    *num_devices = 0;
  else {
    auto devs = getRuntime().getDevices();
    *num_devices = devs->count();
  }
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
  auto devs = getRuntime().getDevices();
  auto device = devs->object<MTL::Device>(device_id);
  if (!device)
    return -1; // Invalid device_id

  switch (property_id) {
    case NP_Name: {
      std::string name = device->name()->cString(NS::StringEncoding::ASCIIStringEncoding);
      if (property_value != NULL) {
        strncpy((char*)property_value, name.c_str(), name.size());
      } else if (property_value_size != NULL) {
        *property_value_size = name.size();
      }
      break;
    }
    case NP_Architecture: {
      auto arch = device->architecture();
      std::string name = arch->name()->cString(NS::StringEncoding::ASCIIStringEncoding);
      if (property_value != NULL) {
        strncpy((char*)property_value, name.c_str(), name.size());
      } else if (property_value_size != NULL) {
        *property_value_size = name.size();
      }
      break;
    }
    default:
      return -1; // Invalid property_id
  }
  return NXS_SUCCESS;
}
