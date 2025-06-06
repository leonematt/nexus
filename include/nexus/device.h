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
    public:
        Device(detail::Impl base);
        using Object::Object;
        
        void release() const;

        nxs_int getId() const override;

        // Get Device Property Value
        template <typename T>
        const T getProperty(nxs_property pn) const;
        
        Properties getProperties() const;

        // Runtime functions
        Librarys getLibraries() const;
        Schedules getSchedules() const;

        Schedule createSchedule();

        Library createLibrary(void *libraryData, size_t librarySize);
        Library createLibrary(const std::string &libraryPath);

        Buffer copyBuffer(Buffer buf);
    };
    
    typedef Objects<Device> Devices;

}

#endif // NEXUS_DEVICE_H
