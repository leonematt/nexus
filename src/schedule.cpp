
#include <nexus/schedule.h>
#include <nexus/command.h>
#include <nexus/log.h>

#include "_device_impl.h"

#define NEXUS_LOG_MODULE "schedule"

using namespace nexus;
using namespace nexus::detail;

namespace nexus {
namespace detail {

  typedef DevObject<Command> DevCommand;

  class ScheduleImpl : public Schedule::OwnerRef {
  public:
    /// @brief Construct a Platform for the current system
    ScheduleImpl(Schedule::OwnerRef owner)
      : OwnerRef(owner) {
        NEXUS_LOG(NEXUS_STATUS_NOTE, "  Schedule: " << getId());
      }

    ~ScheduleImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Schedule: " << getId());
      release();
    }

    void release() {
      //getOwner()->releaseSchedule(getId());
    }

    Command getCommand(Kernel kern) {
      nxs_int kid = kern.getDevId();
      auto cid = getOwner()->createCommand(getId(), kid);
      Command cmd(Command::OwnerRef(this, commands.size()), kern);
      commands.emplace_back(cmd, cid);
      return cmd;
    }

  private:
    std::vector<DevCommand> commands;
};
}
}


///////////////////////////////////////////////////////////////////////////////
Schedule::Schedule(OwnerRef owner) : Object(owner) {}

//Schedule::Schedule() : Object() {}

void Schedule::release() const {
  get()->release();
}

nxs_int Schedule::getId() const {
  return get()->getId();
}

Command Schedule::createCommand(Kernel kern) {
  return get()->getCommand(kern);
}