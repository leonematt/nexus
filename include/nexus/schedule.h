#ifndef NEXUS_SCHEDULE_H
#define NEXUS_SCHEDULE_H

#include <nexus/object.h>
#include <nexus/command.h>
#include <nexus-api.h>

#include <list>

namespace nexus {
    
    namespace detail {
        class DeviceImpl; // owner
        class ScheduleImpl;
    }

    // System class
    class Schedule : public Object<detail::ScheduleImpl, detail::DeviceImpl> {
        friend OwnerTy;
    public:
        Schedule(detail::Impl owner);
        using Object::Object;

        void release() const;

        nxs_int getId() const override;

        Command createCommand(Kernel kern);

        nxs_status run(nxs_bool blocking = true);
    };
}

#endif // NEXUS_SCHEDULE_H