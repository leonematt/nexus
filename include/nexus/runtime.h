#ifndef NEXUS_RUNTIME_H
#define NEXUS_RUNTIME_H

#include <nexus/device.h>
#include <nexus-api.h>

#include <optional>
#include <string>
#include <memory>

namespace nexus {

    namespace detail {
        class RuntimeImpl {
        public:
            RuntimeImpl(const std::string &path);
            ~RuntimeImpl();

            void release();

            int getDeviceCount() const;

            Device getDevice(nxs_int deviceId);

            template <typename T>
            T getFunction(nxs_function fn) const { return (T)runtimeFns[fn]; }

            // Get Runtime Property Value
            template <typename T>
            const T getProperty(nxs_property pn) const {
                size_t size = sizeof(T);
                T val = 0;
                //assert(typeid(T), typeid(pm_t)); // how to lookup at runtime
                if (auto fn = getFunction<nxsGetRuntimeProperty_fn>(FN_nxsGetRuntimeProperty))
                    (*fn)(pn, &val, &size);
                return val;
            }
            template <>
            const std::string getProperty<std::string>(nxs_property pn) const;

            // Get Device Property Value
            template <typename T>
            const T getProperty(nxs_uint deviceId, nxs_property pn) const {
                size_t size = sizeof(T);
                T val = 0;
                //assert(typeid(T), typeid(pm_t)); // how to lookup at runtime
                if (auto fn = getFunction<nxsGetDeviceProperty_fn>(FN_nxsGetDeviceProperty))
                    (*fn)(deviceId, pn, &val, &size);
                return val;
            }
            template <>
            const std::string getProperty<std::string>(nxs_uint deviceId, nxs_property pn) const;

        private:
            void loadPlugin();

            std::string pluginLibraryPath;
            void * library;
            void * runtimeFns[NXSAPI_FUNCTION_COUNT];

            std::vector<Device> devices;
        };
    }

    // Runtime class
    class Runtime : Object<detail::RuntimeImpl> {
    public:
        using Object::Object;

        void release() { return get()->release(); }

        int getDeviceCount() const { return get()->getDeviceCount(); }
        Device getDevice(nxs_uint deviceId) { return get()->getDevice(deviceId); }

        // Get Runtime Property Value
        template <typename T>
        const T getProperty(nxs_property pn) const {
            return get()->getProperty<T>(pn);
        }
        // Get Device Property Value
        template <typename T>
        const T getProperty(nxs_uint deviceId, nxs_property pn) const {
            return get()->getProperty<T>(deviceId, pn);
        }
    };
    
}

#endif // NEXUS_RUNTIME_H
