#ifndef NEXUS_LIBRARY_H
#define NEXUS_LIBRARY_H

#include <nexus/object.h>
#include <nexus-api.h>

#include <list>

namespace nexus {
    
    namespace detail {
        class DeviceImpl;
        class LibraryImpl;
    }

    // System class
    class Library : Object<detail::LibraryImpl> {
        friend detail::DeviceImpl;
    public:
    Library(detail::DeviceImpl *_dev, void *_hostData);
    Library(detail::DeviceImpl *_dev, const std::string &_filePath);
    using Object::Object;

        void release() const;
        
        size_t getSize() const;
        void *getHostData() const;

    protected:
        void _addDevice(Device _dev);
    };
}

#endif // NEXUS_LIBRARY_H