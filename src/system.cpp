
#include <nexus/log.h>
#include <nexus/system.h>
#include <nexus/utility.h>

#include "_system_impl.h"

using namespace nexus;
using namespace nexus::detail;

#define NEXUS_LOG_MODULE "system"

/// @brief Construct a Platform for the current system
SystemImpl::SystemImpl(int) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "CTOR");
  iterateEnvPaths("NEXUS_RUNTIME_PATH", "./runtime_libs",
                  [&](const std::string &path, const std::string &name) {
                    Runtime rt(detail::Impl(this, runtimes.size()), path);
                    runtimes.add(rt);
                  });
}

SystemImpl::~SystemImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "DTOR");
  // for (auto rt : runtimes)
  //   rt.release();
  // for (auto buf : buffers)
  //   buf.release();
}

std::optional<Property> SystemImpl::getProperty(nxs_int prop) const {
  return std::nullopt;
}

Buffer SystemImpl::createBuffer(size_t sz, const void *hostData) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "createBuffer " << sz);
  nxs_uint id = buffers.size();
  Buffer buf(detail::Impl(this, id), sz, hostData);
  buffers.add(buf);
  return buf;
}

Buffer SystemImpl::copyBuffer(Buffer buf, Device dev) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "copyBuffer " << buf.getSize());
  Buffer nbuf = dev.copyBuffer(buf);
  return nbuf;
}

///////////////////////////////////////////////////////////////////////////////
/// @param
System::System(int i) : Object(i) {}

std::optional<Property> System::getProperty(nxs_int prop) const {
  return get()->getProperty(prop);
}

Buffers System::getBuffers() const { return get()->getBuffers(); }

Runtimes System::getRuntimes() const { return get()->getRuntimes(); }

Runtime System::getRuntime(int idx) const { return get()->getRuntime(idx); }

Buffer System::createBuffer(size_t sz, const void *hostData) {
  return get()->createBuffer(sz, hostData);
}

Buffer System::copyBuffer(Buffer buf, Device dev) {
  return get()->copyBuffer(buf, dev);
}

/// @brief Get the System Platform
/// @return
nexus::System nexus::getSystem() {
  static System s_system(0);
  return s_system;
}
