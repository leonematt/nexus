#ifndef _NEXUS_LIBRARY_IMPL_H
#define _NEXUS_LIBRARY_IMPL_H

#include <nexus/library.h>
#include <nexus/kernel.h>

#include "_device_impl.h"

namespace nexus {
namespace detail {

  typedef DevObject<Kernel> DevKernel;

  class LibraryImpl : public Library::OwnerRef {
  public:
    /// @brief Construct a Platform for the current system
    LibraryImpl(Library::OwnerRef owner);

    ~LibraryImpl();

    void release();

    Kernel getKernel(const std::string &kernelName);

    nxs_int getKernelDevId(nxs_int k);

  private:
    std::vector<DevKernel> kernels;
};
}
}

#endif // _NEXUS_LIBRARY_IMPL_H