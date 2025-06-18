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
    BufferImpl(Impl base, nxs_int _devId, size_t _sz, void *_hostData);

    ~BufferImpl();

    void release();

    nxs_int getDeviceId() const { return deviceId; }

    std::optional<Property> getProperty(nxs_int prop) const;
    
    size_t getSize() const { return size; }
    void *getHostData() const { return data; }

    Buffer getLocal() const;
    nxs_status copyData(void *_hostBuf) const;

  private:
    // set of runtimes
    nxs_int deviceId;
    size_t size;
    void *data;
  };
}
}

#endif // _NEXUS_BUFFER_IMPL_H