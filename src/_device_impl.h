#ifndef _NEXUS_DEVICE_IMPL_H
#define _NEXUS_DEVICE_IMPL_H

#include <nexus/properties.h>
#include <nexus/buffer.h>
#include <nexus/library.h>
#include <nexus/runtime.h>

#include "_runtime_impl.h"

namespace nexus {
namespace detail {

  struct DevBuffer {
    Buffer buf;
    nxs_int devId; // from runtime
    DevBuffer(Buffer _b, nxs_int _id) : buf(_b), devId(_id) {}
  };
  struct DevLibrary {
    Library lib;
    nxs_int devId; // from runtime
    DevLibrary(Library _l, nxs_int _id) : lib(_l), devId(_id) {}
  };

  /// @class DesignImpl
  class DeviceImpl : OwnerRef<RuntimeImpl> {
    Properties deviceProps;
    std::vector<DevBuffer> buffers;
    std::vector<nxs_uint> queues;
    std::vector<DevLibrary> libraries;
  public:
    DeviceImpl(OwnerRef base);
    ~DeviceImpl();

    void release();

    Properties getProperties() const { return deviceProps; }

    // Runtime functions
    nxs_int createCommandList();

    Library createLibrary(const std::string &path);
    Library createLibrary(void *libraryData, size_t size);
    nxs_status releaseLibrary(nxs_int lid);

    nxs_status _copyBuffer(Buffer buf);
  };

} // namespace detail
} // namespace nexus

#endif // _NEXUS_DESIGN_IMPL_H