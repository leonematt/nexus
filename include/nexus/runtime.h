#ifndef NEXUS_RUNTIME_H
#define NEXUS_RUNTIME_H

#include <nexus/device.h>
#include <nexus-api.h>

#include <optional>
#include <string>
#include <memory>

namespace nexus {
    class Runtime;

    namespace detail {
        class RuntimeImpl {
        public:
            RuntimeImpl(const std::string &path);
            ~RuntimeImpl();

            int getDeviceCount() const {
                return localDevices.size();
            }

            Device getDevice(nxs_int deviceId) {
                if (deviceId < 0 || deviceId >= localDevices.size())
                    return Device();
                return localDevices[deviceId];
            }

            template <typename T>
            T getFunction(NXSAPI_FunctionEnum fn) const { return (T)runtimeFns[fn]; }

            // Get Runtime Property Value
            template <typename T>
            const T getProperty(NXSAPI_PropertyEnum pn) const {
                size_t size = sizeof(T);
                T val = 0;
                //assert(typeid(T), typeid(pm_t)); // how to lookup at runtime
                if (auto fn = getFunction<nxsGetRuntimeProperty_fn>(FN_nxsGetRuntimeProperty))
                    (*fn)(pn, &val, &size);
                return val;
            }
            template <>
            const std::string getProperty<std::string>(NXSAPI_PropertyEnum pn) const {
                if (auto fn = getFunction<nxsGetRuntimeProperty_fn>(FN_nxsGetRuntimeProperty)) {
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
                if (auto fn = getFunction<nxsGetDeviceProperty_fn>(FN_nxsGetDeviceProperty))
                    (*fn)(deviceId, pn, &val, &size);
                return val;
            }
            template <>
            const std::string getProperty<std::string>(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const {
                if (auto fn = getFunction<nxsGetDeviceProperty_fn>(FN_nxsGetDeviceProperty)) {
                    size_t size = 256;
                    char name[size];
                    (*fn)(deviceId, pn, name, &size);
                    return name;
                }
                return std::string();
            }

        private:
            void loadPlugin();

            std::string pluginLibraryPath;
            void * library;
            void * runtimeFns[NXSAPI_FUNCTION_COUNT];

            std::vector<Device> localDevices;
        };
    }

    // Runtime class
    class Runtime : Object<detail::RuntimeImpl> {
    public:
        using Object::Object;

        int getDeviceCount() const { return get()->getDeviceCount(); }
        Device getDevice(nxs_uint deviceId) { return get()->getDevice(deviceId); }

        // Get Runtime Property Value
        template <typename T>
        const T getProperty(NXSAPI_PropertyEnum pn) const {
            return get()->getProperty<T>(pn);
        }
        // Get Device Property Value
        template <typename T>
        const T getProperty(nxs_uint deviceId, NXSAPI_PropertyEnum pn) const {
            return get()->getProperty<T>(deviceId, pn);
        }
    };
    
}

#endif // NEXUS_RUNTIME_H
