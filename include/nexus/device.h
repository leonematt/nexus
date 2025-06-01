#ifndef NEXUS_DEVICE_H
#define NEXUS_DEVICE_H

#include <nexus/buffer.h>
#include <nexus/library.h>
#include <nexus/schedule.h>
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
    class Device : public Object<detail::DeviceImpl, detail::RuntimeImpl> {
        friend OwnerTy;
        friend detail::SystemImpl;
    public:
        Device(OwnerRef base);
        using Object::Object;
        
        void release() const;

        nxs_int getId() const override;

        Properties getProperties() const;

        // Runtime functions
        Schedule createSchedule();

        Library createLibrary(void *libraryData, size_t librarySize);
        Library createLibrary(const std::string &libraryPath);

    protected:
        nxs_status _copyBuffer(Buffer buf);
    };
    
}

#endif // NEXUS_DEVICE_H
