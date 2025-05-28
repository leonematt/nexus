#ifndef NEXUS_BUFFER_H
#define NEXUS_BUFFER_H

#include <nexus/runtime.h>

#include <vector>

namespace nexus {

    namespace detail {
        class SystemImpl;

        class BufferImpl {
        public:
            BufferImpl(SystemImpl *_sys, size_t _size);
    
        private:
            SystemImpl *system;

            // set of runtimes
            size_t size;
            void *data;
            Runtime runtime;
        };
    
    }

    // System class
    class Buffer : Object<detail::BufferImpl> {
    public:

    };
}

#endif // NEXUS_BUFFER_H