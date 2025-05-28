#ifndef NEXUS_RUNTIME_H
#define NEXUS_RUNTIME_H

#include <nexus/properties.h>
#include <nexus-api.h>

#include <optional>
#include <string>
#include <memory>

namespace nexus {

    class Runtime {
        std::string pluginLibraryPath;
        void * library;
        void * runtimeFns[NXSAPI_FUNCTION_COUNT];

    public:
        // RTDevice - wrapper for Device properties and Runtime actions
        class RTDevice {
            Runtime &runtime;
            nxs_uint id;
            Properties deviceProps;
            std::vector<nxs_uint> buffers;
            std::vector<nxs_uint> queues;
        public:
            RTDevice(Runtime &rt, nxs_uint id);

            Properties getProperties() const { return deviceProps; }

            // Runtime functions
            nxs_int createBuffer(size_t size, void *host_data = nullptr);
            nxs_int createCommandList();

        };
    private:
        std::vector<RTDevice> localDevices;

        void *getRuntimeFunc(NXSAPI_FunctionEnum fn) const { return runtimeFns[fn]; }

    public:
        Runtime(const std::string &path);
        ~Runtime();

        int getDeviceCount() const {
            return localDevices.size();
        }

        RTDevice *getDevice(nxs_uint deviceId) {
            if (deviceId >= localDevices.size())
                return nullptr;
            return &localDevices[deviceId];
        }

        // Get Runtime Property Value
        template <typename T>
        const T getProperty(NXSAPI_PropertyEnum pn) const {
            size_t size = sizeof(T);
            T val = 0;
            //assert(typeid(T), typeid(pm_t)); // how to lookup at runtime
            if (auto fn = (nxsGetRuntimeProperty_fn)runtimeFns[FN_nxsGetRuntimeProperty])
                (*fn)(pn, &val, &size);
            return val;
        }
        template <>
        const std::string getProperty<std::string>(NXSAPI_PropertyEnum pn) const {
            if (auto fn = (nxsGetRuntimeProperty_fn)runtimeFns[FN_nxsGetRuntimeProperty]) {
                size_t size = 256;
                char name[size];
                (*fn)(pn, name, &size);
                return name;
            }
            return std::string();
        }
        // Get Device Property Value
        template <typename T>
        const T getProperty(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const {
            size_t size = sizeof(T);
            T val = 0;
            //assert(typeid(T), typeid(pm_t)); // how to lookup at runtime
            if (auto fn = (nxsGetDeviceProperty_fn)runtimeFns[FN_nxsGetDeviceProperty])
                (*fn)(deviceId, pn, &val, &size);
            return val;
        }
        template <>
        const std::string getProperty<std::string>(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const {
            if (auto fn = (nxsGetDeviceProperty_fn)runtimeFns[FN_nxsGetDeviceProperty]) {
                size_t size = 256;
                char name[size];
                (*fn)(deviceId, pn, name, &size);
                return name;
            }
            return std::string();
        }

    private:
        void loadPlugin();
    };

}

#endif // NEXUS_RUNTIME_H
