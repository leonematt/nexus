#ifndef NEXUS_COMMAND_H
#define NEXUS_COMMAND_H

#include <nexus/object.h>
#include <nexus/kernel.h>
#include <nexus-api.h>

#include <list>

namespace nexus {
    
    namespace detail {
        class ScheduleImpl; // owner
        class CommandImpl;
    }

    // System class
    class Command : public Object<detail::CommandImpl, detail::ScheduleImpl> {
        friend OwnerTy;
    public:
        Command(OwnerRef owner, Kernel kern);
        using Object::Object;

        void release() const;

        nxs_int getId() const override;
    };
}

#endif // NEXUS_COMMAND_H