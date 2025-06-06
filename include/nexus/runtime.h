#ifndef NEXUS_RUNTIME_H
#define NEXUS_RUNTIME_H

#include <nexus/device.h>

#include <string>

namespace nexus {

    namespace detail {
        class RuntimeImpl;
        class SystemImpl;
    }

    // Runtime class
    class Runtime : public Object<detail::RuntimeImpl, detail::SystemImpl> {
        friend OwnerTy;
    public:
        Runtime(detail::Impl owner, const std::string& libraryPath);
        using Object::Object;

        void release() override;

        nxs_int getId() const override;

        Devices getDevices() const;
        Device getDevice(nxs_uint deviceId) const;

        // Get Runtime Property Value
        template <typename T>
        const T getProperty(nxs_property pn) const;
    };
    
    typedef Objects<Runtime> Runtimes;

}

#endif // NEXUS_RUNTIME_H
