#ifndef NEXUS_DEVICE_H
#define NEXUS_DEVICE_H

#include <nexus/properties.h>
#include <nexus-api.h>

#include <optional>
#include <string>
#include <memory>

namespace nexus {

    namespace detail {
        class RuntimeImpl;
        // RTDevice - wrapper for Device properties and Runtime actions
        class DeviceImpl {
            RuntimeImpl *runtime;
            nxs_uint id;
            Properties deviceProps;
            std::vector<nxs_uint> buffers;
            std::vector<nxs_uint> queues;
        public:
            DeviceImpl(RuntimeImpl *rt, nxs_uint id);

            Properties getProperties() const { return deviceProps; }

            // Runtime functions
            nxs_int createBuffer(size_t size, void *host_data = nullptr);
            nxs_int createCommandList();

        };
    }

    // Device class
    class Device {
        // set of runtimes
        typedef detail::DeviceImpl Impl;
        std::shared_ptr<Impl> impl;
    
    public:
        template <typename... Args>
        Device(Args... args) : impl(std::make_shared<detail::DeviceImpl>(args...)) {}
        Device() = default;

        Properties getProperties() const { return impl->getProperties(); }

        // Runtime functions
        nxs_int createBuffer(size_t size, void *host_data = nullptr) {
            return impl->createBuffer(size, host_data);
        }
        nxs_int createCommandList() {
            return impl->createCommandList();
        }
    };
    
}

#endif // NEXUS_DEVICE_H
