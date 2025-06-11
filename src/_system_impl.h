
#ifndef _NEXUS_SYSTEM_IMPL_H
#define _NEXUS_SYSTEM_IMPL_H

#include <nexus/buffer.h>
#include <nexus/runtime.h>

namespace nexus {
namespace detail {

  class SystemImpl : public detail::Impl {
  public:
      SystemImpl(int);
      ~SystemImpl();

      Runtime getRuntime(int idx) const {
          return runtimes.get(idx);
      }
      Buffer createBuffer(size_t sz, void *hostData = nullptr);
      Buffer copyBuffer(Buffer buf, Device dev);

      Runtimes getRuntimes() const {
        return runtimes;
      }

      Buffers getBuffers() const {
        return buffers;
      }

  private:
      // set of runtimes
      Runtimes runtimes;
      Buffers buffers;
  };
} // namespace detail
} // namespace nexus

#endif // _NEXUS_SYSTEM_IMPL_H
