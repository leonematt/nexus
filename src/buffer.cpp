
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
    BufferImpl(SystemImpl *_sys, nxs_int _id, size_t _sz, void *_hostData)
      : system(_sys), id(_id), size(_sz), data(_hostData) {
        NEXUS_LOG(NEXUS_STATUS_NOTE, "  Buffer: " << id << " - " << size);
      }

    ~BufferImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Buffer: " << id);
    }

    void release() {
      devices.clear();
    }

    size_t getSize() const { return size; }
    void *getHostData() const { return data; }

    void _addDevice(Device _dev) {
      devices.push_back(_dev);
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

Buffer::Buffer(detail::SystemImpl *_sys, nxs_int _id, size_t _sz, void *_hostData)
  : Object(_sys, _id, _sz, _hostData) {}

void Buffer::release() const {
  get()->release();
}

size_t Buffer::getSize() const {
  return get()->getSize();
}
void *Buffer::getHostData() const {
  return get()->getHostData();
}

void Buffer::_addDevice(Device _dev) {
  get()->_addDevice(_dev);
}
