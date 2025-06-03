
#include <nexus/system.h>
#include <nexus/utility.h>
#include <nexus/log.h>

using namespace nexus;
using namespace nexus::detail;

#define NEXUS_LOG_MODULE "system"

namespace nexus {
namespace detail {

  class SystemImpl : public detail::Impl {
  public:
      SystemImpl(int);
      ~SystemImpl();

      nxs_int getRuntimeCount() const {
        return runtimes.size();
      }
      Runtime getRuntime(int idx) const {
          return runtimes[idx];
      }
      Buffer createBuffer(size_t sz, void *hostData = nullptr);
      Buffer copyBuffer(Buffer buf, Device dev);

  private:
      // set of runtimes
      std::vector<Runtime> runtimes;
      std::vector<Buffer> buffers;
  };
} // namespace detail
} // namespace nexus

/// @brief Construct a Platform for the current system
SystemImpl::SystemImpl(int) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "CTOR");
  iterateEnvPaths("NEXUS_RUNTIME_PATH", "./runtime_libs", [&](const std::string &path, const std::string &name) {
    runtimes.emplace_back(detail::Impl(this, runtimes.size()), path);
  });
}

SystemImpl::~SystemImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "DTOR");
  for (auto rt : runtimes)
    rt.release();
  //for (auto buf : buffers)
  //  buf.release();
}

Buffer SystemImpl::createBuffer(size_t sz, void *hostData) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "createBuffer " << sz);
  nxs_uint id = buffers.size();
  buffers.emplace_back(detail::Impl(this, id), sz, hostData);
  return buffers.back();
}

Buffer SystemImpl::copyBuffer(Buffer buf, Device dev) {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "copyBuffer " << buf.getSize());
  buf._addDevice(dev);
  Buffer nbuf = dev.copyBuffer(buf);
  return nbuf;
}

///////////////////////////////////////////////////////////////////////////////
/// @param  
System::System(int i) : Object(i) {}

nxs_int System::getRuntimeCount() const {
  return get()->getRuntimeCount();
}

Runtime System::getRuntime(int idx) const {
  return get()->getRuntime(idx);
}

Buffer System::createBuffer(size_t sz, void *hostData) {
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

