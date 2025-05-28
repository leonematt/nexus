#ifndef NEXUS_DEVICE_H
#define NEXUS_DEVICE_H

#include <nexus/object.h>
#include <nexus/properties.h>
#include <nexus-api.h>

#include <optional>
#include <string>

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
    class Device : Object<detail::DeviceImpl> {
    public:
        using Object::Object;

        Properties getProperties() const { return get()->getProperties(); }

        // Runtime functions
        nxs_int createBuffer(size_t size, void *host_data = nullptr) {
            return get()->createBuffer(size, host_data);
        }
        nxs_int createCommandList() {
            return get()->createCommandList();
        }
    };
    
}

#endif // NEXUS_DEVICE_H
