#ifndef NEXUS_BUFFER_H
#define NEXUS_BUFFER_H

#include <nexus/runtime.h>

#include <vector>

namespace nexus {
    class Buffer;

    namespace detail {

        class BufferImpl {
        public:
            BufferImpl();
    
        private:
            // set of runtimes
            void *data;
            Runtime runtime;
        };
    
    }

    // System class
    class Buffer {
        // set of runtimes
        typedef detail::BufferImpl Impl;
        std::shared_ptr<Impl> impl;
    
    public:
        Buffer() : impl(std::make_shared<Impl>()) {}

    };
}

#endif // NEXUS_BUFFER_H