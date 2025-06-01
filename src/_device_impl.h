#ifndef _NEXUS_DEVICE_IMPL_H
#define _NEXUS_DEVICE_IMPL_H

#include <nexus/properties.h>
#include <nexus/buffer.h>
#include <nexus/library.h>
#include <nexus/schedule.h>
#include <nexus/runtime.h>

#include "_runtime_impl.h"

namespace nexus {
namespace detail {

  template <typename T>
  struct DevObject {
    T obj;
    nxs_int id;
    DevObject(T _obj, nxs_int _id) : obj(_obj), id(_id) {}
  };

  typedef DevObject<Buffer> DevBuffer;
  typedef DevObject<Library> DevLibrary;
  typedef DevObject<Schedule> DevSchedule;
  
  /// @class DesignImpl
  class DeviceImpl : public OwnerRef<RuntimeImpl> {
    Properties deviceProps;
    std::vector<DevBuffer> buffers;
    std::vector<DevLibrary> libraries;
    std::vector<DevSchedule> schedules;
  public:
    DeviceImpl(OwnerRef base);
    ~DeviceImpl();

    void release();

    Properties getProperties() const { return deviceProps; }

    // Runtime functions
    Schedule createSchedule();

    Library createLibrary(const std::string &path);
    Library createLibrary(void *libraryData, size_t size);
    nxs_status releaseLibrary(nxs_int lid);

    nxs_int getKernel(nxs_int lid, const std::string &kernelName);

    nxs_int createCommand(nxs_int sid, nxs_int kid);

    nxs_status _copyBuffer(Buffer buf);
  };

} // namespace detail
} // namespace nexus

#endif // _NEXUS_DESIGN_IMPL_H