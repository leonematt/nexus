
#ifndef _NEXUS_SCHEDULE_IMPL_H
#define _NEXUS_SCHEDULE_IMPL_H

#include "_device_impl.h"

namespace nexus {
namespace detail {

    class ScheduleImpl : public Impl {
    public:
      /// @brief Construct a Platform for the current system
      ScheduleImpl(Impl owner);
      ~ScheduleImpl();

      void release();

      Command getCommand(Kernel kern);

      nxs_status run(nxs_bool blocking);

    private:
      Objects<Command> commands;
    };
}
}

#endif // _NEXUS_SCHEDULE_IMPL_H
