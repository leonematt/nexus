#include <nexus/device_db.h>
#include <nexus/runtime.h>
#include <nexus/log.h>

#include <assert.h>

#include <dlfcn.h>

using namespace nexus;

#define NEXUS_LOG_MODULE "runtime"


/// @brief Construct a Runtime for the current system
detail::RuntimeImpl::RuntimeImpl(const std::string &path) : pluginLibraryPath(path), library(nullptr) {
  loadPlugin();
}

detail::RuntimeImpl::~RuntimeImpl() {
  if (library != nullptr)
    dlclose(library);
}


void detail::RuntimeImpl::loadPlugin() {
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



