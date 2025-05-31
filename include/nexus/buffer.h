#ifndef NEXUS_BUFFER_H
#define NEXUS_BUFFER_H

#include <nexus/object.h>
#include <nexus-api.h>

#include <list>

namespace nexus {
    class Device;
    
    namespace detail {
        class SystemImpl;
        class BufferImpl;
    }

    // System class
    class Buffer : Object<detail::BufferImpl> {
        friend detail::SystemImpl;
    public:
        Buffer(detail::SystemImpl *_sys, nxs_int _id, size_t _sz, void *_hostData = nullptr);
        //using Object::Object;

        void release() const;
        
        size_t getSize() const;
        void *getHostData() const;

    protected:
        void _addDevice(Device _dev);
    };
}

#endif // NEXUS_BUFFER_H