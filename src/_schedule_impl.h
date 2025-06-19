
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

  std::optional<Property> getProperty(nxs_int prop) const;

  Command getCommand(Kernel kern);

  nxs_status run(nxs_bool blocking);

 private:
  Objects<Command> commands;
};
}  // namespace detail
}  // namespace nexus

#endif  // _NEXUS_SCHEDULE_IMPL_H
