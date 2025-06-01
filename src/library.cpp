
#include <nexus/log.h>

#include "_library_impl.h"

#define NEXUS_LOG_MODULE "library"

using namespace nexus;
using namespace nexus::detail;

/// @brief Construct a Platform for the current system
LibraryImpl::LibraryImpl(Library::OwnerRef owner)
  : OwnerRef(owner) {
    NEXUS_LOG(NEXUS_STATUS_NOTE, "  Library: " << getId());
  }

LibraryImpl::~LibraryImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Library: " << getId());
  release();
}

void LibraryImpl::release() {
  getOwner()->releaseLibrary(getId());
}

Kernel LibraryImpl::getKernel(const std::string &kernelName) {
  auto kid = getOwner()->getKernel(getId(), kernelName);
  Kernel kern(Kernel::OwnerRef(this, kernels.size()), kernelName);
  kernels.emplace_back(kern, kid);
  return kern;
}

nxs_int LibraryImpl::getKernelDevId(nxs_int kid) {
  if (kid >= 0 && kid < kernels.size())
    return kernels[kid].id;
  return NXS_InvalidKernel;
}

///////////////////////////////////////////////////////////////////////////////
Library::Library(OwnerRef owner) : Object(owner) {}

//Library::Library() : Object() {}

void Library::release() const {
  get()->release();
}

nxs_int Library::getId() const {
  return get()->getId();
}

Kernel Library::getKernel(const std::string &kernelName) {
  return get()->getKernel(kernelName);
}