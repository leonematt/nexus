
#include <nexus/log.h>

#include "_library_impl.h"

#define NEXUS_LOG_MODULE "library"

using namespace nexus;
using namespace nexus::detail;

/// @brief Construct a Platform for the current system
LibraryImpl::LibraryImpl(Impl owner)
  : Impl(owner) {
    NEXUS_LOG(NEXUS_STATUS_NOTE, "  Library: " << getId());
  }

LibraryImpl::~LibraryImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Library: " << getId());
  release();
}

void LibraryImpl::release() {
  kernels.clear();
  auto *rt = getParentOfType<RuntimeImpl>();
  nxs_int kid = rt->runPluginFunction<nxsReleaseLibrary_fn>(NF_nxsReleaseLibrary, getId());
}

Kernel LibraryImpl::getKernel(const std::string &kernelName) {
  auto *rt = getParentOfType<RuntimeImpl>();
  nxs_int kid = rt->runPluginFunction<nxsGetKernel_fn>(NF_nxsGetKernel, getId(), kernelName.c_str());
  Kernel kern(Impl(this, kid), kernelName);
  kernels.emplace_back(kern, kid);
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

Kernel Library::getKernel(const std::string &kernelName) {
  return get()->getKernel(kernelName);
}