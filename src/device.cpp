#include <nexus/buffer.h>
#include <nexus/device_db.h>
#include <nexus/log.h>
#include <nexus/runtime.h>

#include <filesystem>

#include "_buffer_impl.h"
#include "_device_impl.h"
#include "_runtime_impl.h"

#define NEXUS_LOG_MODULE "device"

using namespace nexus;

#define APICALL(FUNC, ...)                                                   \
  nxs_int apiResult = getParent()->runAPIFunction<NF_##FUNC>(__VA_ARGS__);   \
  if (nxs_failed(apiResult))                                                 \
  NEXUS_LOG(NEXUS_STATUS_ERR, " API: " << nxsGetFuncName(NF_##FUNC) << " - " \
                                       << nxsGetStatusName(apiResult))

detail::DeviceImpl::DeviceImpl(detail::Impl base) : detail::Impl(base) {
  auto id = getId();
  auto vendor = getProperty(NP_Vendor);
  auto type = getProperty(NP_Type);
  auto arch = getProperty(NP_Architecture);
  if (!vendor || !type || !arch) return;

  auto devTag = vendor->getValue<NP_Vendor>() + "-" +
                type->getValue<NP_Type>() + "-" +
                arch->getValue<NP_Architecture>();
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    DeviceTag: " << devTag);
  if (auto props = nexus::lookupDeviceInfo(devTag))
    deviceInfo = props;
  else  // load defaults
    NEXUS_LOG(NEXUS_STATUS_ERR, "    Device Properties not found");
}

detail::DeviceImpl::~DeviceImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    ~Device: " << getId());
  release();
}

void detail::DeviceImpl::release() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    release: " << getId());
  buffers.clear();

  libraries.clear();
  schedules.clear();
}

std::optional<Property> detail::DeviceImpl::getProperty(nxs_int prop) const {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "Device.getProperty: " << prop);
  auto *runtime = getParent();
  if (auto fn = runtime->getFunction<NF_nxsGetDeviceProperty>()) {
    auto npt_prop = nxs_property_type_map[prop];
    if (npt_prop == NPT_INT) {
      nxs_long val = 0;
      size_t size = sizeof(val);
      if (nxs_success((*fn)(getId(), prop, &val, &size))) return Property(val);
    } else if (npt_prop == NPT_FLT) {
      nxs_double val = 0.;
      size_t size = sizeof(val);
      if (nxs_success((*fn)(getId(), prop, &val, &size))) return Property(val);
    } else if (npt_prop == NPT_STR) {
      size_t size = 256;
      char name[size];
      name[0] = '\0';
      if (nxs_success((*fn)(getId(), prop, &name, &size)))
        return std::string(name);
    } else {
      NEXUS_LOG(NEXUS_STATUS_ERR,
                "Device.getProperty: Unknown property type for - "
                    << nxsGetPropName(prop));
      // assert(0);
    }
  }
  return std::nullopt;
}

// Runtime functions
Library detail::DeviceImpl::createLibrary(void *data, size_t size) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createLibrary");
  APICALL(nxsCreateLibrary, getId(), data, size);
  Library lib(detail::Impl(this, apiResult));
  libraries.add(lib);
  return lib;
}

Library detail::DeviceImpl::createLibrary(const std::string &path) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createLibrary");
  APICALL(nxsCreateLibraryFromFile, getId(), path.c_str());
  Library lib(detail::Impl(this, apiResult));
  libraries.add(lib);
  return lib;
}

Schedule detail::DeviceImpl::createSchedule() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createSchedule");
  APICALL(nxsCreateSchedule, getId(), 0);
  Schedule sched(detail::Impl(this, apiResult));
  schedules.add(sched);
  return sched;
}

Buffer detail::DeviceImpl::createBuffer(size_t size, const char *data) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createBuffer");
  APICALL(nxsCreateBuffer, getId(), size, 0, (void *)data);
  Buffer nbuf(Impl(this, apiResult), getId(), size);
  buffers.add(nbuf);
  return nbuf;
}

Buffer detail::DeviceImpl::copyBuffer(Buffer buf) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  copyBuffer");
  APICALL(nxsCreateBuffer, getId(), buf.getSize(), 0, (void *)buf.getData());
  Buffer nbuf(Impl(this, apiResult), getId(), buf.getSize());
  buffers.add(nbuf);
  return nbuf;
}

///////////////////////////////////////////////////////////////////////////////
/// Object wrapper - Device
///////////////////////////////////////////////////////////////////////////////
Device::Device(detail::Impl base) : Object(base) {}

nxs_int Device::getId() const { return get()->getId(); }

// Get Device Property Value
std::optional<Property> Device::getProperty(nxs_int prop) const {
  return get()->getProperty(prop);
}

Properties Device::getInfo() const { return get()->getInfo(); }

// Runtime functions
Librarys Device::getLibraries() const { return get()->getLibraries(); }

Schedules Device::getSchedules() const { return get()->getSchedules(); }

Schedule Device::createSchedule() { return get()->createSchedule(); }

Buffer Device::createBuffer(size_t size, const void *data) {
  return get()->createBuffer(size, (const char *)data);
}

Buffer Device::copyBuffer(Buffer buf) { return get()->copyBuffer(buf); }

Library Device::createLibrary(void *libraryData, size_t librarySize) {
  return get()->createLibrary(libraryData, librarySize);
}

Library Device::createLibrary(const std::string &libraryPath) {
  return get()->createLibrary(libraryPath);
}
