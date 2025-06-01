
#include <nexus/buffer.h>
#include <nexus/system.h>
#include <nexus/log.h>

using namespace nexus;
using namespace nexus::detail;

#define NEXUS_LOG_MODULE "buffer"

namespace nexus {
namespace detail {
  class BufferImpl : public Buffer::OwnerRef {
  public:
    //BufferImpl(SystemImpl *_sys, size_t _size, void *_hostData = nullptr);
    /// @brief Construct a Platform for the current system
    BufferImpl(Buffer::OwnerRef base, size_t _sz, void *_hostData)
      : OwnerRef(base), size(_sz), data(_hostData) {
        NEXUS_LOG(NEXUS_STATUS_NOTE, "  Buffer: " << getId() << " - " << size);
      }

    ~BufferImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Buffer: " << getId());
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
    // set of runtimes
    size_t size;
    void *data;
    std::list<Device> devices;
  };
}
}

///////////////////////////////////////////////////////////////////////////////
Buffer::Buffer(OwnerRef base, size_t _sz, void *_hostData)
  : Object(base, _sz, _hostData) {}

void Buffer::release() const {
  get()->release();
}

nxs_int Buffer::getId() const {
  return get()->getId();
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
