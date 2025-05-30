#ifndef NEXUS_DEVICE_H
#define NEXUS_DEVICE_H

#include <nexus/buffer.h>
#include <nexus/properties.h>
#include <nexus-api.h>

#include <optional>
#include <string>

namespace nexus {

    namespace detail {
        class SystemImpl;
        class RuntimeImpl;
        class DeviceImpl;
    }

    // Device class
    class Device : Object<detail::DeviceImpl> {
        friend detail::SystemImpl;
    public:
        Device(detail::RuntimeImpl *rt, nxs_uint id);
        Device();
        
        void release() const;

        Properties getProperties() const;

        // Runtime functions
        nxs_int createCommandList();
        
    protected:
        nxs_status _copyBuffer(Buffer buf);
    };
    
}

#endif // NEXUS_DEVICE_H
