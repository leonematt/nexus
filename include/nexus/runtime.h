#ifndef NEXUS_RUNTIME_H
#define NEXUS_RUNTIME_H

#include <nexus/device.h>
#include <nexus-api.h>

#include <string>

namespace nexus {

    namespace detail {
        class RuntimeImpl;
    }

    // Runtime class
    class Runtime : Object<detail::RuntimeImpl> {
    public:
        Runtime(const std::string& libraryPath);
        using Object::Object;

        void release();

        int getDeviceCount() const;
        Device getDevice(nxs_uint deviceId);

        // Get Runtime Property Value
        template <typename T>
        const T getProperty(nxs_property pn) const;
        // Get Device Property Value
        template <typename T>
        const T getProperty(nxs_uint deviceId, nxs_property pn) const;
    };
    
}

#endif // NEXUS_RUNTIME_H
