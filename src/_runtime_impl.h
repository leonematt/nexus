#ifndef _NEXUS_RUNTIME_IMPL_H
#define _NEXUS_RUNTIME_IMPL_H

#include <nexus/device.h>
#include <nexus-api.h>

#include <optional>
#include <string>
#include <memory>

#define NEXUS_LOG_MODULE "runtime"

namespace nexus {

    namespace detail {
        class RuntimeImpl : public Runtime::OwnerRef {
        public:
            RuntimeImpl(Runtime::OwnerRef owner, const std::string &path);
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

            template <typename Tfn, typename... Args>
            nxs_int runPluginFunction(nxs_function fne, Args... args) {
                nxs_int apiResult = NXS_InvalidDevice;
                if (auto *fn = getFunction<Tfn>(fne)) {
                  apiResult = (*fn)(args...);
                  NEXUS_LOG(NEXUS_STATUS_NOTE, nxsGetFuncName(fne) << ": " << apiResult);
                } else {
                  NEXUS_LOG(NEXUS_STATUS_ERR, nxsGetFuncName(fne) << ": API not present");
                }
                return apiResult;
            }

        private:
            void loadPlugin();

            std::string pluginLibraryPath;
            void * library;
            void * runtimeFns[NXSAPI_FUNCTION_COUNT];

            std::vector<Device> devices;
        };

    }
}

#undef NEXUS_LOG_MODULE

#endif // _NEXUS_RUNTIME_IMPL_H
