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
        Buffer(detail::Impl base, nxs_int devId, size_t _sz, void *_hostData = nullptr);
        using Object::Object;

        void release() const;
        
        nxs_int getId() const override;
        nxs_int getDeviceId() const;

        size_t getSize() const;
        void *getHostData() const;

        Buffer getLocal() const;

        nxs_status copy(void *_hostBuf);
    };

    typedef Objects<Buffer> Buffers;
}

#endif // NEXUS_BUFFER_H