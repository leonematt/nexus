
#ifndef _NEXUS_SCHEDULE_IMPL_H
#define _NEXUS_SCHEDULE_IMPL_H

#include "_device_impl.h"

namespace nexus {
namespace detail {

    typedef DevObject<Command> DevCommand;

    class ScheduleImpl : public Schedule::OwnerRef {
    public:
      /// @brief Construct a Platform for the current system
      ScheduleImpl(Schedule::OwnerRef owner);
      ~ScheduleImpl();

      void release();

      Command getCommand(Kernel kern);

    private:
      std::vector<DevCommand> commands;
    };
}
}

#endif // _NEXUS_SCHEDULE_IMPL_H
