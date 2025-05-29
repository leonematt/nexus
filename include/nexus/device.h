#ifndef NEXUS_DEVICE_H
#define NEXUS_DEVICE_H

#include <nexus/buffer.h>
#include <nexus/properties.h>
#include <nexus-api.h>

#include <optional>
#include <string>

namespace nexus {

    namespace detail {
        class RuntimeImpl;
        class DeviceImpl;
    }

    // Device class
    class Device : Object<detail::DeviceImpl> {
    public:
        Device(detail::RuntimeImpl *rt, nxs_uint id);
        Device();
        
        Properties getProperties() const;

        // Runtime functions
        nxs_int createBuffer(size_t size, void *hostData = nullptr);
        nxs_int createCommandList();
    };
    
}

#endif // NEXUS_DEVICE_H
