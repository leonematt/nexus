#ifndef _NEXUS_DEVICE_IMPL_H
#define _NEXUS_DEVICE_IMPL_H

#include <nexus/properties.h>
#include <nexus/buffer.h>
#include <nexus/runtime.h>

#include "_runtime_impl.h"

namespace nexus {
namespace detail {

  struct DeviceBuf {
    Buffer buf;
    nxs_uint devId;
    DeviceBuf(Buffer _b, nxs_uint _id) : buf(_b), devId(_id) {}
  };

  /// @class DesignImpl
  class DeviceImpl {
    RuntimeImpl *runtime;
    nxs_uint id;
    Properties deviceProps;
    std::vector<DeviceBuf> buffers;
    std::vector<nxs_uint> queues;
  public:
    DeviceImpl(RuntimeImpl *rt, nxs_uint id);
    ~DeviceImpl();

    void release();

    Properties getProperties() const { return deviceProps; }

    // Runtime functions
    nxs_int createBuffer(size_t size, void *hostData = nullptr);
    nxs_int createCommandList();
    nxs_int createLibrary(void *data, size_t size);
    nxs_int createLibrary(const std::string &path);

    nxs_status _copyBuffer(Buffer buf);

  };

} // namespace detail
} // namespace nexus

#endif // _NEXUS_DESIGN_IMPL_H