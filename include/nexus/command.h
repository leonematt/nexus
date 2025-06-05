#ifndef NEXUS_COMMAND_H
#define NEXUS_COMMAND_H

#include <nexus/object.h>
#include <nexus/kernel.h>
#include <nexus/buffer.h>
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
        Command(detail::Impl owner, Kernel kern);
        using Object::Object;

        void release() const;
        nxs_int getId() const override;

        nxs_status setArgument(nxs_uint index, Buffer buffer) const;
        //nxs_status setArgument(nxs_uint index, nxs_int scalar) const;

        nxs_status finalize(nxs_int gridSize, nxs_int groupSize);
    };

    typedef Objects<Command> Commands;

}

#endif // NEXUS_COMMAND_H