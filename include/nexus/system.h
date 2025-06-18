#ifndef NEXUS_SYSTEM_H
#define NEXUS_SYSTEM_H

#include <nexus/runtime.h>
#include <nexus/buffer.h>

#include <vector>
#include <optional>
#include <memory>

namespace nexus {
    namespace detail {
        class SystemImpl;
    }

    // System class
    class System : Object<detail::SystemImpl> {
    public:
        System(int);
        using Object::Object;
    
        nxs_int getId() const override { return 0; }

        std::optional<Property> getProperty(nxs_int prop) const override;

        Runtimes getRuntimes() const;
        Buffers getBuffers() const;

        Runtime getRuntime(int idx) const;
        Buffer createBuffer(size_t sz, void *hostData = nullptr);
        Buffer copyBuffer(Buffer buf, Device dev);
    };

    extern System getSystem();
}

#endif // NEXUS_SYSTEM_H