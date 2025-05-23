#ifndef NEXUS_RUNTIME_H
#define NEXUS_RUNTIME_H

#include <nexus/device.h>
#include <nexus-api/nxs_runtime.h>

#include <optional>
#include <string>
#include <memory>

namespace nexus {

    enum RuntimeFn {
        nxsGetRuntimeProperty,
        nxsGetDeviceProperty,
        RuntimeFnSize
    };

    class Runtime {
        std::string pluginLibraryPath;
        void * library;
        void * runtimeFns[RuntimeFn::RuntimeFnSize];
        std::vector<Device> localDevices;
    public:
        Runtime(const std::string &path);
        ~Runtime();

        std::string getName() const;

    private:
        void loadPlugin();
    };

}

#endif // NEXUS_RUNTIME_H
