
#include <nexus/kernel.h>
#include <nexus/log.h>

#include "_library_impl.h"

#define NEXUS_LOG_MODULE "kernel"

using namespace nexus;
using namespace nexus::detail;

namespace nexus {
namespace detail {
  class KernelImpl : public Kernel::OwnerRef {
  public:
    /// @brief Construct a Platform for the current system
    KernelImpl(Kernel::OwnerRef owner, const std::string &kName)
      : OwnerRef(owner), kernelName(kName) {
        NEXUS_LOG(NEXUS_STATUS_NOTE, "  Kernel: " << kernelName << " - " << getId());
      }

    ~KernelImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Kernel: " << getId());
      release();
    }

    void release() {
      //getOwner()->releaseKernel(getId());
    }

  private:
    std::string kernelName;
  };
}
}


///////////////////////////////////////////////////////////////////////////////
Kernel::Kernel(OwnerRef owner, const std::string &kernelName)
  : Object(owner, kernelName) {}

void Kernel::release() const {
  get()->release();
}

nxs_int Kernel::getId() const {
  return get()->getId();
}
