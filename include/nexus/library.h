#ifndef NEXUS_LIBRARY_H
#define NEXUS_LIBRARY_H

#include <nexus/object.h>
#include <nexus-api.h>

#include <list>

namespace nexus {
    
    namespace detail {
        class DeviceImpl; // owner
        class LibraryImpl;
    }

    // System class
    class Library : Object<detail::LibraryImpl> {
        friend detail::DeviceImpl;
    public:
        Library(detail::DeviceImpl *_dev, nxs_int id);
        Library();
        //using Object::Object;

        void release() const;
        
        size_t getSize() const;
        void *getHostData() const;
    };
}

#endif // NEXUS_LIBRARY_H