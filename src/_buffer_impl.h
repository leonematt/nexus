#ifndef _NEXUS_BUFFER_IMPL_H
#define _NEXUS_BUFFER_IMPL_H

#include <nexus/device.h>

namespace nexus {
namespace detail {
  class BufferImpl : public Impl {
  public:
    //BufferImpl(SystemImpl *_sys, size_t _size, void *_hostData = nullptr);
    /// @brief Construct a Platform for the current system
    BufferImpl(Impl base, size_t _sz, void *_hostData);

    ~BufferImpl();

    void release();

    size_t getSize() const { return size; }
    void *getHostData() const { return data; }

    nxs_status copyData(void *_hostBuf);

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

#endif // _NEXUS_BUFFER_IMPL_H