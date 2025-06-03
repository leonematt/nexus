
#include <nexus/command.h>
#include <nexus/log.h>

#include "_schedule_impl.h"

#define NEXUS_LOG_MODULE "command"

using namespace nexus;

namespace nexus {
namespace detail {
  class CommandImpl : public Impl {
  public:
    /// @brief Construct a Platform for the current system
    CommandImpl(Impl owner, Kernel kern)
      : Impl(owner) {
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
      //arguments[index] = buffer;
      auto *rt = getParentOfType<RuntimeImpl>();
      return (nxs_status)rt->runPluginFunction<nxsSetCommandArgument_fn>(FN_nxsSetCommandArgument, getId(), index, buffer.getId());
    }

    nxs_status finalize(nxs_int groupSize, nxs_int gridSize) {
      //arguments[index] = buffer;
      auto *rt = getParentOfType<RuntimeImpl>();
      return (nxs_status)rt->runPluginFunction<nxsFinalizeCommand_fn>(FN_nxsFinalizeCommand, getId(), groupSize, gridSize);
    }

  private:
    std::vector<Buffer> arguments;
  };
}
}


///////////////////////////////////////////////////////////////////////////////
Command::Command(detail::Impl owner, Kernel kern)
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

nxs_status Command::finalize(nxs_int groupSize, nxs_int gridSize) {
  return get()->finalize(groupSize, gridSize);
}
