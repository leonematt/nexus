#ifndef NEXUS_KERNEL_H
#define NEXUS_KERNEL_H

#include <nexus-api.h>
#include <nexus/object.h>

#include <list>

namespace nexus {

namespace detail {
class LibraryImpl;  // owner
class KernelImpl;
}  // namespace detail

// System class
class Kernel : public Object<detail::KernelImpl, detail::LibraryImpl> {
  friend OwnerTy;

 public:
  Kernel(detail::Impl owner, const std::string &kernelName);
  using Object::Object;

  void release() const;
  nxs_int getId() const override;

  std::optional<Property> getProperty(nxs_int prop) const override;
};

typedef Objects<Kernel> Kernels;

}  // namespace nexus

#endif  // NEXUS_KERNEL_H