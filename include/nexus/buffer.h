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
    public:
        Buffer(detail::SystemImpl *_sys, nxs_uint _id, size_t _sz, void *_hostData = nullptr);
        using Object::Object;

        nxs_int copyToDevice(Device _dev);
    };
}

#endif // NEXUS_BUFFER_H