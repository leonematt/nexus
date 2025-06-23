#ifndef _NEXUS_BUFFER_IMPL_H
#define _NEXUS_BUFFER_IMPL_H

#include <nexus/device.h>

namespace nexus {
namespace detail {
class BufferImpl : public Impl {
 public:
  typedef std::shared_ptr<std::vector<char>> DataBuf;

  BufferImpl(Impl base, size_t _sz, const char *_hostData);
  BufferImpl(Impl base, nxs_int _devId, size_t _sz, const char *_hostData);

  ~BufferImpl();

  void release();

  nxs_int getDeviceId() const { return deviceId; }

  std::optional<Property> getProperty(nxs_int prop) const;

  size_t getSize() const { return data ? data->size() : 0; }
  const char *getData() const { return data ? data->data() : nullptr; }

  void setData(size_t sz, const char *hostData);
  void setData(DataBuf _data) { data = _data; }

  Buffer getLocal() const;
  nxs_status copyData(void *_hostBuf) const;

  std::string print() const;

 private:
  // set of runtimes
  nxs_int deviceId;
  DataBuf data;
};
}  // namespace detail
}  // namespace nexus

#endif  // _NEXUS_BUFFER_IMPL_H