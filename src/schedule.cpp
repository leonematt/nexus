
#include <nexus/schedule.h>
#include <nexus/command.h>
#include <nexus/log.h>

#include "_schedule_impl.h"

#define NEXUS_LOG_MODULE "schedule"

using namespace nexus;
using namespace nexus::detail;

/// @brief Construct a Platform for the current system
ScheduleImpl::ScheduleImpl(detail::Impl owner)
  : detail::Impl(owner) {
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
  auto *rt = getParentOfType<RuntimeImpl>();
  nxs_int cid = rt->runPluginFunction<nxsCreateCommand_fn>(NF_nxsCreateCommand, getId(), kern.getId());
  Command cmd(detail::Impl(this, cid), kern);
  commands.emplace_back(cmd, cid);
  return cmd;
}

nxs_status ScheduleImpl::run(nxs_bool blocking) {
  auto *rt = getParentOfType<RuntimeImpl>();
  return (nxs_status)rt->runPluginFunction<nxsRunSchedule_fn>(NF_nxsRunSchedule, getId(), blocking);
}

///////////////////////////////////////////////////////////////////////////////
Schedule::Schedule(detail::Impl owner) : Object(owner) {}

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

nxs_status Schedule::run(nxs_bool blocking) {
  return get()->run(blocking);
}
