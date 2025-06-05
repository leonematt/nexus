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

  /// @class DesignImpl
  class DeviceImpl : public Impl {
    Properties deviceProps;
    Buffers buffers;
    Objects<Library> libraries;
    Objects<Schedule> schedules;
  public:
    DeviceImpl(Impl base);
    virtual ~DeviceImpl();

    void release();

    RuntimeImpl *getParent() const { return Impl::getParent<RuntimeImpl>(); }

    // Get Runtime Property Value
    template <typename T>
    const T getProperty(nxs_property pn) const {
      size_t size = sizeof(T);
      T val = 0;
      if (auto fn = getParent()->getFunction<NF_nxsGetDeviceProperty>())
        (*fn)(getId(), pn, &val, &size);
      return val;
    }
    template <>
    const std::string getProperty<std::string>(nxs_property pn) const;

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