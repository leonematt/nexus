
#include <nexus/kernel.h>
#include <nexus/log.h>

#include "_library_impl.h"

#define NEXUS_LOG_MODULE "kernel"

using namespace nexus;

namespace nexus {
namespace detail {
class KernelImpl : public Impl {
 public:
  /// @brief Construct a Platform for the current system
  KernelImpl(Impl owner, const std::string &kName)
      : Impl(owner), kernelName(kName) {
    NEXUS_LOG(NEXUS_STATUS_NOTE,
              "  Kernel: " << kernelName << " - " << getId());
  }

  ~KernelImpl() {
    NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Kernel: " << getId());
    release();
  }

  void release() {
    // getOwner()->releaseKernel(getId());
  }

  std::optional<Property> getProperty(nxs_int prop) const {
    return std::nullopt;
  }

 private:
  std::string kernelName;
};
}  // namespace detail
}  // namespace nexus

///////////////////////////////////////////////////////////////////////////////
Kernel::Kernel(detail::Impl owner, const std::string &kernelName)
    : Object(owner, kernelName) {}

void Kernel::release() const { get()->release(); }

nxs_int Kernel::getId() const { return get()->getId(); }

std::optional<Property> Kernel::getProperty(nxs_int prop) const {
  return get()->getProperty(prop);
}
