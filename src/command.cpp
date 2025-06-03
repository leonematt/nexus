
#include <nexus/command.h>
#include <nexus/log.h>

#include "_schedule_impl.h"

#define NEXUS_LOG_MODULE "command"

using namespace nexus;
using namespace nexus::detail;

namespace nexus {
namespace detail {
  class CommandImpl : public Command::OwnerRef {
  public:
    /// @brief Construct a Platform for the current system
    CommandImpl(Command::OwnerRef owner, Kernel kern)
      : OwnerRef(owner) {
        NEXUS_LOG(NEXUS_STATUS_NOTE, "    Command: " << getId());

        //TODO: gather kernel argument details
      }

    ~CommandImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "    ~Command: " << getId());
      release();
    }

    void release() {
      //getOwner()->releaseCommand(getId());
    }

    nxs_status setArgument(nxs_uint index, Buffer buffer) {
      arguments[index] = buffer;
      auto *rt = getParentOfType<RuntimeImpl>();
      return (nxs_status)rt->runPluginFunction<nxsSetCommandArgument_fn>(FN_nxsSetCommandArgument, getId(), index, buffer.getId());
    }

  private:
    std::vector<Buffer> arguments;
  };
}
}


///////////////////////////////////////////////////////////////////////////////
Command::Command(OwnerRef owner, Kernel kern)
  : Object(owner, kern) {}

void Command::release() const {
  get()->release();
}

nxs_int Command::getId() const {
  return get()->getId();
}

nxs_status Command::setArgument(nxs_uint index, Buffer buffer) const {
  return get()->setArgument(index, buffer);
}
        