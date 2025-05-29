
#include <nexus/buffer.h>
#include <nexus/system.h>
#include <nexus/log.h>

using namespace nexus;
using namespace nexus::detail;

#define NEXUS_LOG_MODULE "buffer"

namespace nexus {
namespace detail {
  class BufferImpl {
  public:
    //BufferImpl(SystemImpl *_sys, size_t _size, void *_hostData = nullptr);
    /// @brief Construct a Platform for the current system
    BufferImpl(SystemImpl *_sys, nxs_uint _id, size_t _sz, void *_hostData)
      : system(_sys), id(_id), size(_sz), data(_hostData) {
        NEXUS_LOG(NEXUS_STATUS_NOTE, "  Buffer: " << id << " - " << size);
      }

    ~BufferImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Buffer: " << id);
    }
    nxs_int copyToDevice(Device _dev) {
      devices.push_back(_dev);
      return NXS_Success;
    }
  private:
    SystemImpl *system;
    nxs_uint id;

    // set of runtimes
    size_t size;
    void *data;
    std::list<Device> devices;
  };
}
}

Buffer::Buffer(detail::SystemImpl *_sys, nxs_uint _id, size_t _sz, void *_hostData)
  : Object(_sys, _id, _sz, _hostData) {}

nxs_int Buffer::copyToDevice(Device _dev) {
  return get()->copyToDevice(_dev);
}
