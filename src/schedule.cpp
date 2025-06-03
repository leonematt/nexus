
#include <nexus/schedule.h>
#include <nexus/command.h>
#include <nexus/log.h>

#include "_schedule_impl.h"

#define NEXUS_LOG_MODULE "schedule"

using namespace nexus;
using namespace nexus::detail;

/// @brief Construct a Platform for the current system
ScheduleImpl::ScheduleImpl(Schedule::OwnerRef owner)
  : OwnerRef(owner) {
    NEXUS_LOG(NEXUS_STATUS_NOTE, "  Schedule: " << getId());
  }

ScheduleImpl::~ScheduleImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Schedule: " << getId());
  release();
}

void ScheduleImpl::release() {
  //getOwner()->releaseSchedule(getId());
}

Command ScheduleImpl::getCommand(Kernel kern) {
  nxs_int kid = kern.getId();
  auto cid = getParent<Schedule::OwnerTy>()->createCommand(getId(), kid);
  Command cmd(Command::OwnerRef(this, commands.size()), kern);
  commands.emplace_back(cmd, cid);
  return cmd;
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