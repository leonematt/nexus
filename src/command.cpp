
#include <nexus/command.h>
#include <nexus/log.h>

#include "_device_impl.h"

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
        NEXUS_LOG(NEXUS_STATUS_NOTE, "  Command: " << " - " << getId());
      }

    ~CommandImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Command: " << getId());
      release();
    }

    void release() {
      //getOwner()->releaseCommand(getId());
    }

  private:
  };
}
}


///////////////////////////////////////////////////////////////////////////////
Command::Command(OwnerRef owner, Kernel kern)
  : Object(owner, kern) {}

//Command::Command() : Object() {}

void Command::release() const {
  get()->release();
}

nxs_int Command::getId() const {
  return get()->getId();
}

