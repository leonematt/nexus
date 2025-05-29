
#include <nexus/system.h>
#include <nexus/utility.h>
#include <nexus/log.h>

using namespace nexus;
using namespace nexus::detail;

#define NEXUS_LOG_MODULE "system"

namespace nexus {
namespace detail {

  class SystemImpl {
  public:
      SystemImpl(int);
      ~SystemImpl();

      Runtime getRuntime(int idx) const {
          return runtimes[idx];
      }
      Buffer createBuffer(size_t sz, void *hostData = nullptr);
      nxs_status copyBuffer(Buffer buf, Device dev);

  private:
      // set of runtimes
      std::vector<Runtime> runtimes;
      std::vector<Buffer> buffers;
  };
} // namespace detail
} // namespace nexus

/// @brief Construct a Platform for the current system
SystemImpl::SystemImpl(int) {
  iterateEnvPaths("NEXUS_RUNTIME_PATH", "./runtime_libs", [&](const std::string &path, const std::string &name) {
    runtimes.emplace_back(path);
  });
}

SystemImpl::~SystemImpl() {
  NEXUS_LOG(NEXUS_STATUS_NOTE, "~System: ");
}

Buffer SystemImpl::createBuffer(size_t sz, void *hostData) {
  nxs_uint id = buffers.size();
  buffers.emplace_back(this, id, sz, hostData);
  return buffers.back();
}

nxs_status SystemImpl::copyBuffer(Buffer buf, Device dev) {
  //dev->_copyBuffer(buf);
  return NXS_Success;
}

///////////////////////////////////////////////////////////////////////////////
/// @param  
System::System(int i) : Object(i) {}

Runtime System::getRuntime(int idx) const {
  return get()->getRuntime(idx);
}

Buffer System::createBuffer(size_t sz, void *hostData) {
  return get()->createBuffer(sz, hostData);
}


/// @brief Get the System Platform
/// @return 
nexus::System nexus::getSystem() {
  static System s_system(0);
  return s_system;
}

