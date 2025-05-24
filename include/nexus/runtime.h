#ifndef NEXUS_RUNTIME_H
#define NEXUS_RUNTIME_H

#include <nexus/device.h>
#include <nexus-api.h>

#include <optional>
#include <string>
#include <memory>

namespace nexus {

    class Runtime {
        std::string pluginLibraryPath;
        void * library;
        void * runtimeFns[NXSAPI_FUNCTION_COUNT];
        std::vector<Device> localDevices;
    public:
        Runtime(const std::string &path);
        ~Runtime();

        int getDeviceCount() const;

        std::string getStrProperty(NXSAPI_PropertyEnum pn) const;
        const nxs_uint getIntProperty(NXSAPI_PropertyEnum pn) const;
        const nxs_double getFloatProperty(NXSAPI_PropertyEnum pn) const;

        std::string getStrProperty(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const;
        const nxs_uint getIntProperty(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const;
        const nxs_double getFloatProperty(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const;

    private:
        void loadPlugin();
    };

}

#endif // NEXUS_RUNTIME_H
