
#include <nexus/log.h>

#include "_library_impl.h"

#define NEXUS_LOG_MODULE "library"
#undef NEXUS_LOG_DEPTH
#define NEXUS_LOG_DEPTH 34

using namespace nexus;
using namespace nexus::detail;

/// @brief Construct a Platform for the current system
LibraryImpl::LibraryImpl(Impl owner)
  : Impl(owner) {
    NEXUS_LOG(NEXUS_STATUS_NOTE, "CTOR: " << getId());
  }

LibraryImpl::~LibraryImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "DTOR: " << getId());
  release();
}

void LibraryImpl::release() {
  kernels.clear();
  auto *rt = getParentOfType<RuntimeImpl>();
  nxs_int kid = rt->runAPIFunction<NF_nxsReleaseLibrary>(getId());
}

std::optional<Property> detail::LibraryImpl::getProperty(nxs_int prop) const {
  return std::nullopt;
}

Kernel LibraryImpl::getKernel(const std::string &kernelName) {
  auto *rt = getParentOfType<RuntimeImpl>();
  nxs_int kid = rt->runAPIFunction<NF_nxsGetKernel>(getId(), kernelName.c_str());
  Kernel kern(Impl(this, kid), kernelName);
  kernels.add(kern);
  return kern;
}

///////////////////////////////////////////////////////////////////////////////
Library::Library(Impl owner) : Object(owner) {}

void Library::release() const {
  get()->release();
}

nxs_int Library::getId() const {
  return get()->getId();
}

std::optional<Property> Library::getProperty(nxs_int prop) const {
  return get()->getProperty(prop);
}

Kernel Library::getKernel(const std::string &kernelName) {
  return get()->getKernel(kernelName);
}