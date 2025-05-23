#ifndef NEXUS_SYSTEM_H
#define NEXUS_SYSTEM_H

#include <nexus/runtime.h>

#include <vector>
#include <optional>
#include <memory>

namespace nexus {

    // Platform class
    class System {
        // set of runtimes
        typedef std::shared_ptr<Runtime> RuntimePtr;
        std::vector<RuntimePtr> runtimes;
    
    public:
        System();

        RuntimePtr getRuntime(int idx) const {
            return runtimes[idx];
        }
    private:
    };

    System &getSystem();
}

#endif // NEXUS_SYSTEM_H