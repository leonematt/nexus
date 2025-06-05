#include <nexus/device_db.h>
#include <nexus/runtime.h>
#include <nexus/buffer.h>
#include <nexus/log.h>

#include "_runtime_impl.h"
#include "_device_impl.h"
#include "_buffer_impl.h"

#define NEXUS_LOG_MODULE "device"

using namespace nexus;

#define APICALL(FUNC, ...) \
  nxs_int apiResult = getParent()->runPluginFunction<FUNC##_fn>(NF_##FUNC, __VA_ARGS__); \
  if (apiResult < 0) \
    NEXUS_LOG(NEXUS_STATUS_ERR, " API: " << nxsGetFuncName(NF_##FUNC) << " - " << nxsGetStatusName((nxs_status)apiResult))
    

detail::DeviceImpl::DeviceImpl(detail::Impl base)
: detail::Impl(base) {
  auto id = getId();
  auto vendor = getProperty<std::string>(NP_Vendor);
  auto type = getProperty<std::string>(NP_Type);
  auto arch = getProperty<std::string>(NP_Architecture);
  auto devTag = vendor + "-" + type + "-" + arch;
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    DeviceTag: " << devTag);
  if (auto props = nexus::lookupDevice(devTag))
    deviceProps = *props;
  else // load defaults
    NEXUS_LOG(NEXUS_STATUS_ERR, "    Device Properties not found");
}

detail::DeviceImpl::~DeviceImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    ~Device: " << getId());
  release();
}

void detail::DeviceImpl::release() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    release: " << getId());
  for (auto &buf : buffers) {
    // release from device
    //buf.buf.release(id); // not owned
  }
  buffers.clear();

  libraries.clear();
  schedules.clear();
}

template <>
const std::string detail::DeviceImpl::getProperty<std::string>(nxs_property pn) const {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "Device.getProperty: " << pn);
  auto *runtime = getParent();
  if (auto fn = runtime->getFunction<nxsGetDeviceProperty_fn>(NF_nxsGetDeviceProperty)) {
      size_t size = 256;
      char name[size];
      name[0] = '\0';
      (*fn)(getId(), pn, &name, &size);
      return name;
  }
  return std::string();
}


Library detail::DeviceImpl::createLibrary(void *data, size_t size) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createLibrary");
  APICALL(nxsCreateLibrary, getId(), data, size);
  Library lib(detail::Impl(this, apiResult));
  libraries.emplace_back(lib, apiResult);
  return lib;
}

Library detail::DeviceImpl::createLibrary(const std::string &path) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createLibrary");
  APICALL(nxsCreateLibraryFromFile, getId(), path.c_str());
  Library lib(detail::Impl(this, apiResult));
  libraries.emplace_back(lib, apiResult);
  return lib;
}

Schedule detail::DeviceImpl::createSchedule() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createSchedule");
  APICALL(nxsCreateSchedule, getId(), 0);
  Schedule sched(detail::Impl(this, apiResult));
  schedules.emplace_back(sched, apiResult);
  return sched;
}

Buffer detail::DeviceImpl::copyBuffer(Buffer buf) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  copyBuffer");
  APICALL(nxsCreateBuffer, getId(), buf.getSize(), 0, buf.getHostData());
  Buffer nbuf(Impl(this, apiResult), buf.getSize());
  buffers.emplace_back(nbuf, apiResult);
  return nbuf;
}

///////////////////////////////////////////////////////////////////////////////
/// @return 
///////////////////////////////////////////////////////////////////////////////
//Device::Device(detail::RuntimeImpl *rt, nxs_uint id) : Object(rt, id) {}
Device::Device(detail::Impl base) : Object(base) {}

void Device::release() const {
  get()->release();
}

nxs_int Device::getId() const {
  return get()->getId();
}


// Get Device Property Value
template <>
const std::string Device::getProperty<std::string>(nxs_property pn) const {
    return get()->getProperty<std::string>(pn);
}
template <>
const int64_t Device::getProperty<int64_t>(nxs_property pn) const {
    return get()->getProperty<int64_t>(pn);
}
template <>
const double Device::getProperty<double>(nxs_property pn) const {
    return get()->getProperty<double>(pn);
}

Properties Device::getProperties() const { return get()->getProperties(); }

// Runtime functions
Schedule Device::createSchedule() {
    return get()->createSchedule();
}

Buffer Device::copyBuffer(Buffer buf) {
  return get()->copyBuffer(buf);
}

Library Device::createLibrary(void *libraryData, size_t librarySize) {
  return get()->createLibrary(libraryData, librarySize);
}

Library Device::createLibrary(const std::string &libraryPath) {
  return get()->createLibrary(libraryPath);
}
