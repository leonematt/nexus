
#include <nexus/buffer.h>
#include <nexus/system.h>
#include <nexus/log.h>

#include "_buffer_impl.h"
#include "_runtime_impl.h"

#define NEXUS_LOG_MODULE "buffer"

using namespace nexus;

detail::BufferImpl::BufferImpl(detail::Impl base, size_t _sz, void *_hostData)
: Impl(base), size(_sz), data(_hostData) {

}

detail::BufferImpl::~BufferImpl() {
  release();
}

void detail::BufferImpl::release() {

}

nxs_status detail::BufferImpl::copyData(void *_hostBuf) {
  auto *rt = getParentOfType<RuntimeImpl>();
  return (nxs_status)rt->runPluginFunction<nxsCopyBuffer_fn>(NF_nxsCopyBuffer, getId(), _hostBuf);
}


///////////////////////////////////////////////////////////////////////////////
Buffer::Buffer(detail::Impl base, size_t _sz, void *_hostData)
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

nxs_status Buffer::copy(void *_hostBuf) {
  return get()->copyData(_hostBuf);
}

void Buffer::_addDevice(Device _dev) {
  get()->_addDevice(_dev);
}
