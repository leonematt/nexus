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
            SystemImpl(int);
    
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
    class System : Object<detail::SystemImpl> {
    public:
        using Object::Object;
    
        Runtime getRuntime(int idx) const {
            return get()->getRuntime(idx);
        }
    };

    extern System getSystem();
}

#endif // NEXUS_SYSTEM_H