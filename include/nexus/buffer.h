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
    class Buffer : public Object<detail::BufferImpl, detail::SystemImpl> {
        friend OwnerTy;
    public:
        Buffer(detail::Impl base, size_t _sz, void *_hostData = nullptr);
        using Object::Object;

        void release() const;
        
        nxs_int getId() const override;

        size_t getSize() const;
        void *getHostData() const;

        nxs_status copy(void *_hostBuf);

    protected:
        void _addDevice(Device _dev);
    };

    typedef Objects<Buffer> Buffers;
}

#endif // NEXUS_BUFFER_H