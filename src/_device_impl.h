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
  class DeviceImpl : public Impl {
    Properties deviceProps;
    std::vector<DevBuffer> buffers;
    std::vector<DevLibrary> libraries;
    std::vector<DevSchedule> schedules;
  public:
    DeviceImpl(Impl base);
    virtual ~DeviceImpl();

    void release();

    RuntimeImpl *getParent() const { return Impl::getParent<RuntimeImpl>(); }

    Properties getProperties() const { return deviceProps; }

    // Runtime functions

    Library createLibrary(const std::string &path);
    Library createLibrary(void *libraryData, size_t size);
    Schedule createSchedule();

    Buffer copyBuffer(Buffer buf);
  };

} // namespace detail
} // namespace nexus

#endif // _NEXUS_DESIGN_IMPL_H