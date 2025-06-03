
#ifndef _NEXUS_SCHEDULE_IMPL_H
#define _NEXUS_SCHEDULE_IMPL_H

#include "_device_impl.h"

namespace nexus {
namespace detail {

    typedef DevObject<Command> DevCommand;

    class ScheduleImpl : public Impl {
    public:
      /// @brief Construct a Platform for the current system
      ScheduleImpl(Impl owner);
      ~ScheduleImpl();

      void release();

      Command getCommand(Kernel kern);

      nxs_status run(nxs_bool blocking);

    private:
      std::vector<DevCommand> commands;
    };
}
}

#endif // _NEXUS_SCHEDULE_IMPL_H
