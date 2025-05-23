#include <nexus/runtime.h>
#include <nexus/log.h>

#include <dlfcn.h>

#include <nexus-api/nxs_function_types.h>

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
  library = dlopen(pluginLibraryPath.c_str(), RTLD_NOW);
  if (library == nullptr) {
    NEXUS_LOG(NEXUS_STATUS_ERR, "  Failed to load plugin");
    assert(0);
  }

  auto loadFn = [&](RuntimeFn fn, const char *name) {
    runtimeFns[fn] = dlsym(library, name);
    assert(runtimeFns[fn] != nullptr);
  };

  loadFn(RuntimeFn::nxsGetRuntimeProperty, "nxsGetRuntimeProperty");
}

std::string Runtime::getName() const {
  size_t size = 128;
  char name[size];
  (*(nxsGetRuntimeProperty_fn)runtimeFns[RuntimeFn::nxsGetRuntimeProperty])(NXS_RUNTIME_NAME, name, &size);
  return name;
}
