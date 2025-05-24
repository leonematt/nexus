#include <nexus/runtime.h>
#include <nexus/log.h>

#include <dlfcn.h>

using namespace nexus;

#define NEXUS_LOG_MODULE "runtime"


/// @brief Construct a Platform for the current system
Runtime::Runtime(const std::string &path) : pluginLibraryPath(path), library(nullptr) {
  loadPlugin();
}

Runtime::~Runtime() {
  if (library != nullptr)
    dlclose(library);
}


void Runtime::loadPlugin() {
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

  auto loadFn = [&](NXSAPI_FunctionEnum fn) {
    auto *fName = nxsGetFuncName(fn);
    runtimeFns[fn] = dlsym(library, fName+3);
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

  // Lazy load Device properties
}

std::string Runtime::getStrProperty(NXSAPI_PropertyEnum pn) const {
  auto fn = (nxsGetRuntimeProperty_fn)runtimeFns[FN_nxsGetRuntimeProperty];
  if (!fn)
    return "";
  size_t size = 256;
  char name[size];
  (*fn)(pn, name, &size);
  return name;
}

const nxs_uint Runtime::getIntProperty(NXSAPI_PropertyEnum pn) const {
  auto fn = (nxsGetRuntimeProperty_fn)runtimeFns[FN_nxsGetRuntimeProperty];
  if (!fn)
    return 0;
  size_t size = sizeof(nxs_uint);
  nxs_uint value;
  (*fn)(pn, &value, &size);
  return value;
}

const nxs_double Runtime::getFloatProperty(NXSAPI_PropertyEnum pn) const {
  auto fn = (nxsGetRuntimeProperty_fn)runtimeFns[FN_nxsGetRuntimeProperty];
  if (!fn)
    return 0;
  size_t size = sizeof(nxs_double);
  nxs_double value;
  (*fn)(pn, &value, &size);
  return value;
}

int Runtime::getDeviceCount() const {
  nxs_uint count = 0;
  auto fn = (nxsGetDeviceCount_fn)runtimeFns[FN_nxsGetDeviceCount];
  if (fn)
    (*fn)(NXS_DEVICE_TYPE_GPU, &count);
  return count;
}

std::string Runtime::getStrProperty(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const {
  auto fn = (nxsGetDeviceProperty_fn)runtimeFns[FN_nxsGetDeviceProperty];
  if (!fn)
    return "";
  size_t size = 256;
  char name[size];
  (*fn)(deviceId, pn, name, &size);
  return name;
}

const nxs_uint Runtime::getIntProperty(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const {
  auto fn = (nxsGetDeviceProperty_fn)runtimeFns[FN_nxsGetDeviceProperty];
  if (!fn)
    return 0;
  size_t size = sizeof(nxs_uint);
  nxs_uint value;
  (*fn)(deviceId, pn, &value, &size);
  return value;
}

const nxs_double Runtime::getFloatProperty(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const {
  auto fn = (nxsGetDeviceProperty_fn)runtimeFns[FN_nxsGetDeviceProperty];
  if (!fn)
    return 0;
  size_t size = sizeof(nxs_double);
  nxs_double value;
  (*fn)(deviceId, pn, &value, &size);
  return value;
}

