#include <nexus/device_db.h>
#include <nexus/runtime.h>
#include <nexus/log.h>

#include "_runtime_impl.h"

#include <assert.h>

#include <dlfcn.h>

using namespace nexus;
using namespace nexus::detail;

#define NEXUS_LOG_MODULE "runtime"


/// @brief Construct a Runtime for the current system
RuntimeImpl::RuntimeImpl(const std::string &path) : pluginLibraryPath(path), library(nullptr) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  CTOR: " << path);
  loadPlugin();
}

RuntimeImpl::~RuntimeImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  DTOR: " << pluginLibraryPath);
  release();
  if (library != nullptr)
    dlclose(library);
}

void RuntimeImpl::release() {
  for (auto dev : devices) {
    dev.release();
  }
}

int RuntimeImpl::getDeviceCount() const {
  return devices.size();
}

Device RuntimeImpl::getDevice(nxs_int deviceId) {
  if (deviceId < 0 || deviceId >= devices.size())
      return Device();
  return devices[deviceId];
}


template <>
const std::string RuntimeImpl::getProperty<std::string>(nxs_property pn) const {
  if (auto fn = getFunction<nxsGetRuntimeProperty_fn>(FN_nxsGetRuntimeProperty)) {
      size_t size = 256;
      char name[size];
      (*fn)(pn, name, &size);
      return name;
  }
  return std::string();
}

template <>
const std::string RuntimeImpl::getProperty<std::string>(nxs_uint deviceId, nxs_property pn) const {
  if (auto fn = getFunction<nxsGetDeviceProperty_fn>(FN_nxsGetDeviceProperty)) {
      size_t size = 256;
      char name[size];
      (*fn)(deviceId, pn, name, &size);
      return name;
  }
  return std::string();
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void RuntimeImpl::loadPlugin() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "Loading Runtime plugin: " << pluginLibraryPath);
  library = dlopen(pluginLibraryPath.c_str(), RTLD_NOW | RTLD_GLOBAL);
  char *dlError = dlerror();
  if (dlError) {
    NEXUS_LOG(NEXUS_STATUS_ERR, "  Failed to dlopen plugin: " << dlError);
    assert(0);
  } else if (library == nullptr) {
    NEXUS_LOG(NEXUS_STATUS_ERR, "  Failed to load plugin");
    assert(0);
  }

  auto loadFn = [&](nxs_function fn) {
    auto *fName = nxsGetFuncName(fn);
    runtimeFns[fn] = dlsym(library, fName);
    dlError = dlerror();
    if (dlError) {
      NEXUS_LOG(NEXUS_STATUS_WARN, "  Failed to load symbol '" << fName << "': " << dlError);
    } else {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "  Loaded symbol: " << fName << " - " << (int64_t)runtimeFns[fn]);
    }
  };

  loadFn(FN_nxsGetRuntimeProperty);
  loadFn(FN_nxsGetDeviceCount);
  loadFn(FN_nxsGetDeviceProperty);

  loadFn(FN_nxsCreateBuffer);
  loadFn(FN_nxsReleaseBuffer);

  loadFn(FN_nxsCreateLibrary);
  loadFn(FN_nxsCreateLibraryFromFile);
  loadFn(FN_nxsReleaseLibrary);

  loadFn(FN_nxsCreateCommandList);
  loadFn(FN_nxsRunCommandList);
  loadFn(FN_nxsReleaseCommandList);
  
  loadFn(FN_nxsCreateCommand);
  loadFn(FN_nxsReleaseCommand);

  if (!runtimeFns[FN_nxsGetRuntimeProperty] || !runtimeFns[FN_nxsGetDeviceCount] ||
      !runtimeFns[FN_nxsGetDeviceProperty])
      return;

  // Load device properties

  // Lazy load Device properties
  auto fn = (nxsGetDeviceCount_fn)runtimeFns[FN_nxsGetDeviceCount];
  nxs_int count = (*fn)();
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  DeviceCount - " << count);
  for (int i = 0; i < count; ++i) {
    devices.emplace_back(this, i);
  }
}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
Runtime::Runtime(const std::string &libraryPath) : Object(libraryPath) {
  
}
void Runtime::release() { return get()->release(); }

int Runtime::getDeviceCount() const { return get()->getDeviceCount(); }
Device Runtime::getDevice(nxs_uint deviceId) {
  return get()->getDevice(deviceId);
}

// Get Runtime Property Value
template <>
const std::string Runtime::getProperty<std::string>(nxs_property pn) const {
  return get()->getProperty<std::string>(pn);
}
template <>
const int64_t Runtime::getProperty<int64_t>(nxs_property pn) const {
  return get()->getProperty<int64_t>(pn);
}
template <>
const double Runtime::getProperty<double>(nxs_property pn) const {
  return get()->getProperty<double>(pn);
}

// Get Device Property Value
template <>
const std::string Runtime::getProperty<std::string>(nxs_uint deviceId, nxs_property pn) const {
    return get()->getProperty<std::string>(deviceId, pn);
}
template <>
const int64_t Runtime::getProperty<int64_t>(nxs_uint deviceId, nxs_property pn) const {
    return get()->getProperty<int64_t>(deviceId, pn);
}
template <>
const double Runtime::getProperty<double>(nxs_uint deviceId, nxs_property pn) const {
    return get()->getProperty<double>(deviceId, pn);
}
