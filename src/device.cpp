#include <nexus/device_db.h>
#include <nexus/runtime.h>
#include <nexus/log.h>

#include "_runtime_impl.h"
#include "_device_impl.h"

#define NEXUS_LOG_MODULE "device"

using namespace nexus;
using namespace nexus::detail;

#define APICALL(FUNC, ...) \
  nxs_int apiResult = NXS_InvalidDevice; \
  if (auto fn = runtime->getFunction<FUNC##_fn>(FN_##FUNC)) { \
    apiResult = (*fn)(__VA_ARGS__); \
    NEXUS_LOG(NEXUS_STATUS_NOTE, nxsGetFuncName(FN_##FUNC) << ": " << apiResult); \
  } else { \
    NEXUS_LOG(NEXUS_STATUS_ERR, nxsGetFuncName(FN_##FUNC) << ": API not present"); \
  }


DeviceImpl::DeviceImpl(detail::RuntimeImpl *rt, nxs_uint _id)
: runtime(rt), id(_id) {
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
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    ~Device: " << id);
}

void DeviceImpl::release() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    release: " << id);
  for (auto buf : buffers) {
    // release from device
  }
  buffers.clear();
  queues.clear();
}


nxs_int DeviceImpl::createBuffer(size_t size, void *host_data) {
  APICALL(nxsCreateBuffer, id, size, 0, host_data);
  return apiResult;
}

nxs_int DeviceImpl::createCommandList() {
  APICALL(nxsCreateCommandList, id, 0);
  if (apiResult >= 0) // success
    queues.push_back(apiResult);
  return apiResult;
}

nxs_int DeviceImpl::createLibrary(void *data, size_t size) {
  APICALL(nxsCreateLibrary, id, data, size);
  return apiResult;
}

nxs_int DeviceImpl::createLibrary(const std::string &path) {
  APICALL(nxsCreateLibraryFromFile, id, path.c_str());
  return apiResult;
}

nxs_status DeviceImpl::_copyBuffer(Buffer buf) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  copyBuffer");
  auto bufId = createBuffer(buf.getSize(), buf.getHostData());
  if (bufId > -1) {
    buffers.emplace_back(buf, bufId);
    return NXS_Success;
  }
  return (nxs_status)bufId;
}


///////////////////////////////////////////////////////////////////////////////
/// @return 
///////////////////////////////////////////////////////////////////////////////
Device::Device(detail::RuntimeImpl *rt, nxs_uint id) : Object(rt, id) {}

Device::Device() : Object() {}

void Device::release() const {
  get()->release();
}

Properties Device::getProperties() const { return get()->getProperties(); }

// Runtime functions
nxs_int Device::createCommandList() {
    return get()->createCommandList();
}

nxs_status Device::_copyBuffer(Buffer buf) {
  return get()->_copyBuffer(buf);
}

nxs_int Device::createLibrary(void *libraryData, size_t librarySize) {
  return get()->createLibrary(libraryData, librarySize);
}

nxs_int Device::createLibrary(const std::string &libraryPath) {
  return get()->createLibrary(libraryPath);
}
