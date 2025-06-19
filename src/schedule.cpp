
#include <nexus/command.h>
#include <nexus/log.h>
#include <nexus/schedule.h>

#include "_schedule_impl.h"

#define NEXUS_LOG_MODULE "schedule"

using namespace nexus;
using namespace nexus::detail;

/// @brief Construct a Platform for the current system
ScheduleImpl::ScheduleImpl(detail::Impl owner) : detail::Impl(owner) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  Schedule: " << getId());
}

ScheduleImpl::~ScheduleImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Schedule: " << getId());
  release();
}

void ScheduleImpl::release() {
  // getOwner()->releaseSchedule(getId());
}

std::optional<Property> ScheduleImpl::getProperty(nxs_int prop) const {
  return std::nullopt;
}

Command ScheduleImpl::getCommand(Kernel kern) {
  auto *rt = getParentOfType<RuntimeImpl>();
  nxs_int cid = rt->runAPIFunction<NF_nxsCreateCommand>(getId(), kern.getId());
  Command cmd(detail::Impl(this, cid), kern);
  commands.add(cmd);
  return cmd;
}

nxs_status ScheduleImpl::run(nxs_bool blocking) {
  auto *rt = getParentOfType<RuntimeImpl>();
  return (nxs_status)rt->runAPIFunction<NF_nxsRunSchedule>(getId(), blocking);
}

///////////////////////////////////////////////////////////////////////////////
Schedule::Schedule(detail::Impl owner) : Object(owner) {}

// Schedule::Schedule() : Object() {}

void Schedule::release() const { get()->release(); }

nxs_int Schedule::getId() const { return get()->getId(); }

std::optional<Property> Schedule::getProperty(nxs_int prop) const {
  return get()->getProperty(prop);
}

Command Schedule::createCommand(Kernel kern) { return get()->getCommand(kern); }

nxs_status Schedule::run(nxs_bool blocking) { return get()->run(blocking); }
