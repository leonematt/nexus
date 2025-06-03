#include <nexus/device_db.h>
#include <nexus/runtime.h>
#include <nexus/log.h>

#include "_runtime_impl.h"
#include "_device_impl.h"

#define NEXUS_LOG_MODULE "device"

using namespace nexus;
using namespace nexus::detail;

#define APICALL(FUNC, ...) \
  nxs_int apiResult = getParent()->runPluginFunction<FUNC##_fn>(FN_##FUNC, __VA_ARGS__)


DeviceImpl::DeviceImpl(OwnerRef base)
: OwnerRef(base) {
  auto *runtime = getParent();
  auto id = getId();
  auto vendor = runtime->getProperty<std::string>(id, NP_Vendor);
  auto type = runtime->getProperty<std::string>(id, NP_Type);
  auto arch = runtime->getProperty<std::string>(id, NP_Architecture);
  auto devTag = vendor + "-" + type + "-" + arch;
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    DeviceTag: " << devTag);
  if (auto props = nexus::lookupDevice(devTag))
    deviceProps = *props;
  else // load defaults
    NEXUS_LOG(NEXUS_STATUS_ERR, "    Device Properties not found");
}

DeviceImpl::~DeviceImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    ~Device: " << getId());
  release();
}

void DeviceImpl::release() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    release: " << getId());
  for (auto &buf : buffers) {
    // release from device
    //buf.buf.release(id); // not owned
  }
  buffers.clear();

  libraries.clear();
  schedules.clear();
}

Schedule DeviceImpl::createSchedule() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createSchedule");
  APICALL(nxsCreateSchedule, getId(), 0);
  Schedule sched(Schedule::OwnerRef(this, apiResult));
  schedules.emplace_back(sched, apiResult);
  return sched;
}


nxs_int DeviceImpl::createCommand(nxs_int sid, nxs_int kid) {
  APICALL(nxsCreateCommand, sid, kid);
  return apiResult;
}

Library DeviceImpl::createLibrary(void *data, size_t size) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createLibrary");
  APICALL(nxsCreateLibrary, getId(), data, size);
  Library lib(Library::OwnerRef(this, apiResult));
  libraries.emplace_back(lib, apiResult);
  return lib;
}

Library DeviceImpl::createLibrary(const std::string &path) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  createLibrary");
  APICALL(nxsCreateLibraryFromFile, getId(), path.c_str());
  Library lib(Library::OwnerRef(this, apiResult));
  libraries.emplace_back(lib, apiResult);
  return lib;
}

nxs_status DeviceImpl::_copyBuffer(Buffer buf) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  copyBuffer");
  APICALL(nxsCreateBuffer, getId(), buf.getSize(), 0, buf.getHostData());
  buffers.emplace_back(buf, apiResult);
  return (nxs_status)(apiResult < 0 ? apiResult : NXS_Success);
}

nxs_int DeviceImpl::getKernel(nxs_int lid, const std::string &kernelName) {
  APICALL(nxsGetKernel, lid, kernelName.c_str());
  return apiResult;
}

///////////////////////////////////////////////////////////////////////////////
/// @return 
///////////////////////////////////////////////////////////////////////////////
//Device::Device(detail::RuntimeImpl *rt, nxs_uint id) : Object(rt, id) {}
Device::Device(OwnerRef base) : Object(base) {}

void Device::release() const {
  get()->release();
}

nxs_int Device::getId() const {
  return get()->getId();
}

Properties Device::getProperties() const { return get()->getProperties(); }

// Runtime functions
Schedule Device::createSchedule() {
    return get()->createSchedule();
}

nxs_status Device::_copyBuffer(Buffer buf) {
  return get()->_copyBuffer(buf);
}

Library Device::createLibrary(void *libraryData, size_t librarySize) {
  return get()->createLibrary(libraryData, librarySize);
}

Library Device::createLibrary(const std::string &libraryPath) {
  return get()->createLibrary(libraryPath);
}
