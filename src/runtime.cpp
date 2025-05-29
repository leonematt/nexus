#include <nexus/device_db.h>
#include <nexus/runtime.h>
#include <nexus/log.h>

#include <assert.h>

#include <dlfcn.h>

using namespace nexus;
using namespace nexus::detail;

#define NEXUS_LOG_MODULE "runtime"


/// @brief Construct a Runtime for the current system
RuntimeImpl::RuntimeImpl(const std::string &path) : pluginLibraryPath(path), library(nullptr) {
  loadPlugin();
}

RuntimeImpl::~RuntimeImpl() {
  if (library != nullptr)
    dlclose(library);
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Runtime: " << pluginLibraryPath);  
}

int RuntimeImpl::getDeviceCount() const {
  return localDevices.size();
}

Device RuntimeImpl::getDevice(nxs_int deviceId) {
  if (deviceId < 0 || deviceId >= localDevices.size())
      return Device();
  return localDevices[deviceId];
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
  loadFn(FN_nxsCreateCommandList);
  

  if (!runtimeFns[FN_nxsGetRuntimeProperty] || !runtimeFns[FN_nxsGetDeviceCount] ||
      !runtimeFns[FN_nxsGetDeviceProperty])
      return;

  // Load device properties

  // Lazy load Device properties
  auto fn = (nxsGetDeviceCount_fn)runtimeFns[FN_nxsGetDeviceCount];
  nxs_int count = (*fn)();
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  DeviceCount - " << count);
  for (int i = 0; i < count; ++i) {
    localDevices.emplace_back(this, i);
  }
}
