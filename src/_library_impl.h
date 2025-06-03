#ifndef _NEXUS_LIBRARY_IMPL_H
#define _NEXUS_LIBRARY_IMPL_H

#include <nexus/library.h>
#include <nexus/kernel.h>

#include "_device_impl.h"

namespace nexus {
namespace detail {

  typedef DevObject<Kernel> DevKernel;

  class LibraryImpl : public Impl {
  public:
    /// @brief Construct a Platform for the current system
    LibraryImpl(Impl owner);

    ~LibraryImpl();

    void release();

    Kernel getKernel(const std::string &kernelName);

  private:
    std::vector<DevKernel> kernels;
};
}
}

#endif // _NEXUS_LIBRARY_IMPL_H