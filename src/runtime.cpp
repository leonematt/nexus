#include <nexus/device_db.h>
#include <nexus/runtime.h>
#include <nexus/log.h>

#include <dlfcn.h>

using namespace nexus;

#define NEXUS_LOG_MODULE "runtime"


Runtime::RTDevice::RTDevice(Runtime &rt, nxs_uint _id)
: runtime(rt), id(_id) {
  auto vendor = runtime.getProperty<std::string>(id, NP_Vendor);
  auto type = runtime.getProperty<std::string>(id, NP_Type);
  auto arch = runtime.getProperty<std::string>(id, NP_Architecture);
  auto devTag = vendor + "-" + type + "-" + arch;
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    DeviceTag: " << devTag);
  if (auto props = nexus::lookupDevice(devTag))
    deviceProps = *props;
  else
    NEXUS_LOG(NEXUS_STATUS_ERR, "    Device Properties not found");
}

nxs_int Runtime::RTDevice::createBuffer(size_t size, void *host_data) {
  if (auto fn = (nxsCreateBuffer_fn)runtime.getRuntimeFunc(FN_nxsCreateBuffer)) {
    nxs_int retVal;
    nxs_int bufId = (*fn)(id, size, 0, host_data, &retVal);
    if (retVal == NXS_SUCCESS) {
      buffers.push_back(bufId);
      return bufId;
    }
  }
  return NXS_INVALID_CONTEXT;
}


/// @brief Construct a Runtime for the current system
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

  loadFn(FN_nxsCreateBuffer);

  if (!runtimeFns[FN_nxsGetRuntimeProperty] || !runtimeFns[FN_nxsGetDeviceCount] ||
      !runtimeFns[FN_nxsGetDeviceProperty])
      return;

  // Load device properties

  // Lazy load Device properties
  nxs_uint count = 0;
  auto fn = (nxsGetDeviceCount_fn)runtimeFns[FN_nxsGetDeviceCount];
  (*fn)(&count);
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  DeviceCount - " << count);
  for (int i = 0; i < count; ++i) {
    localDevices.emplace_back(*this, i);
  }
}



