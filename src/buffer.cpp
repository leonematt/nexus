
#include <nexus/buffer.h>
#include <nexus/log.h>
#include <nexus/system.h>

#include <cstring>

#include "_buffer_impl.h"
#include "_runtime_impl.h"
#include "_system_impl.h"

#define NEXUS_LOG_MODULE "buffer"

using namespace nexus;

detail::BufferImpl::BufferImpl(detail::Impl base, size_t _sz, const char *_hostData)
    : Impl(base), deviceId(NXS_InvalidDevice) {
      setData(_sz, _hostData);
    }

detail::BufferImpl::BufferImpl(detail::Impl base, nxs_int _devId, size_t _sz,
                               const char *_hostData)
    : Impl(base), deviceId(_devId) {
      setData(_sz, _hostData);
    }

detail::BufferImpl::~BufferImpl() { release(); }

void detail::BufferImpl::release() {}

std::optional<Property> detail::BufferImpl::getProperty(nxs_int prop) const {
  if (getDeviceId()) {
    //auto *rt = getParentOfType<RuntimeImpl>();
    //if (nxs_success(rt->runAPIFunction<NF_nxsGetBufferProperty>(getId(), prop))) {
    //}
  }
  switch (prop) {
    case NP_Type: return Property("buffer");
    case NP_Size: return Property((nxs_long)data->size());
  }
  return std::nullopt;
}

void detail::BufferImpl::setData(size_t sz, const char *hostData) {
  if (hostData != nullptr) {
    data = std::make_shared<std::vector<char>>();
    data->reserve(sz);
    data->insert(data->end(), &hostData[0], &hostData[sz]);
  } else {
    data = std::make_shared<std::vector<char>>(sz);
  }
}

Buffer detail::BufferImpl::getLocal() const {
  if (!nxs_valid_id(getDeviceId())) return *this;
  void *lbuf = data->data();
  auto *rt = getParentOfType<RuntimeImpl>();
  if (nxs_success(rt->runAPIFunction<NF_nxsCopyBuffer>(getId(), lbuf))) {
    auto *sys = getParentOfType<detail::SystemImpl>();
    return sys->createBuffer(data->size(), data->data());
  }
  return Buffer();
}

nxs_status detail::BufferImpl::copyData(void *_hostBuf) const {
  if (nxs_valid_id(getDeviceId())) {
    auto *rt = getParentOfType<RuntimeImpl>();
    return (nxs_status)rt->runAPIFunction<NF_nxsCopyBuffer>(getId(), _hostBuf);
  }
  memcpy(_hostBuf, getData(), getSize());
  return NXS_Success;
}

///////////////////////////////////////////////////////////////////////////////
Buffer::Buffer(detail::Impl base, size_t _sz, const void *_hostData)
    : Object(base, _sz, (const char *)_hostData) {}

Buffer::Buffer(detail::Impl base, nxs_int _devId, size_t _sz, const void *_hostData)
    : Object(base, _devId, _sz, (const char *)_hostData) {}

nxs_int Buffer::getId() const { return get()->getId(); }

nxs_int Buffer::getDeviceId() const { return get()->getDeviceId(); }

std::optional<Property> Buffer::getProperty(nxs_int prop) const {
  return get()->getProperty(prop);
}

size_t Buffer::getSize() const { return get()->getSize(); }
const char *Buffer::getData() const { return get()->getData(); }

Buffer Buffer::getLocal() const { return get()->getLocal(); }

nxs_status Buffer::copy(void *_hostBuf) { return get()->copyData(_hostBuf); }
