#ifndef NEXUS_DEVICE_H
#define NEXUS_DEVICE_H

#include <nexus/buffer.h>
#include <nexus/library.h>
#include <nexus/properties.h>
#include <nexus-api.h>

#include <optional>
#include <string>

namespace nexus {

    namespace detail {
        class RuntimeImpl; // owner
        class DeviceImpl;
    }

    // Device class
    class Device : Object<detail::DeviceImpl> {
        friend detail::SystemImpl;
    public:
        Device(detail::OwnerRef<detail::RuntimeImpl> base);
        Device();
        
        void release() const;

        Properties getProperties() const;

        // Runtime functions
        nxs_int createCommandList();

        Library createLibrary(void *libraryData, size_t librarySize);
        Library createLibrary(const std::string &libraryPath);

    protected:
        nxs_status _copyBuffer(Buffer buf);
    };
    
}

#endif // NEXUS_DEVICE_H
