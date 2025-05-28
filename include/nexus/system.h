#ifndef NEXUS_SYSTEM_H
#define NEXUS_SYSTEM_H

#include <nexus/runtime.h>

#include <vector>
#include <optional>
#include <memory>

namespace nexus {
    class System;

    namespace detail {

        class SystemImpl {
        public:
            SystemImpl();
    
            Runtime getRuntime(int idx) const {
                return runtimes[idx];
            }
        private:
            // set of runtimes
            std::vector<Runtime> runtimes;
            //std::vector<Buffer> buffers;
        };
    }

    // System class
    class System {
        // set of runtimes
        typedef detail::SystemImpl Impl;
        std::shared_ptr<Impl> impl;
    
    public:
        System() : impl(std::make_shared<Impl>()) {}

        Runtime getRuntime(int idx) const {
            return impl->getRuntime(idx);
        }
    };

    extern System getSystem();
}

#endif // NEXUS_SYSTEM_H