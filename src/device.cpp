#include <nexus/device_db.h>
#include <nexus/runtime.h>
#include <nexus/log.h>

#define NEXUS_LOG_MODULE "device"

using namespace nexus;
using namespace nexus::detail;

namespace nexus {
namespace detail {

  struct DeviceBuf {
    Buffer buf;
    nxs_uint devId;
    DeviceBuf(Buffer _b, nxs_uint _id) : buf(_b), devId(_id) {}
  };

  // RTDevice - wrapper for Device properties and Runtime actions
  class DeviceImpl {
    RuntimeImpl *runtime;
    nxs_uint id;
    Properties deviceProps;
    std::vector<DeviceBuf> buffers;
    std::vector<nxs_uint> queues;
  public:
    DeviceImpl(RuntimeImpl *rt, nxs_uint id);
    ~DeviceImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "    ~Device: " << id);
    }

    void release() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "    release: " << id);
      for (auto buf : buffers) {
        // release from device
      }
      buffers.clear();
      queues.clear();
    }

    Properties getProperties() const { return deviceProps; }

    // Runtime functions
    nxs_int createBuffer(size_t size, void *hostData = nullptr);
    nxs_int createCommandList();

    nxs_status _copyBuffer(Buffer buf) {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "  copyBuffer");
      auto bufId = createBuffer(buf.getSize(), buf.getHostData());
      if (bufId > -1) {
        buffers.emplace_back(buf, bufId);
        return NXS_Success;
      }
      return (nxs_status)bufId;
    }

  };

} // namespace detail
} // namespace nexus

#define APICALL(FUNC, ...) \
  nxs_int fres = NXS_InvalidDevice; \
  if (auto fn = runtime->getFunction<FUNC##_fn>(FN_##FUNC)) { \
    fres = (*fn)(__VA_ARGS__); \
    NEXUS_LOG(NEXUS_STATUS_NOTE, nxsGetFuncName(FN_##FUNC) << ": " << fres); \
  } else { \
    NEXUS_LOG(NEXUS_STATUS_ERR, nxsGetFuncName(FN_##FUNC) << ": API not present"); \
  } \
  return fres


DeviceImpl::DeviceImpl(detail::RuntimeImpl *rt, nxs_uint _id)
: runtime(rt), id(_id) {
  auto vendor = runtime->getProperty<std::string>(id, NP_Vendor);
  auto type = runtime->getProperty<std::string>(id, NP_Type);
  auto arch = runtime->getProperty<std::string>(id, NP_Architecture);
  auto devTag = vendor + "-" + type + "-" + arch;
  NEXUS_LOG(NEXUS_STATUS_NOTE, "    DeviceTag: " << devTag);
  if (auto props = nexus::lookupDevice(devTag))
    deviceProps = *props;
  else // load defaults
    NEXUS_LOG(NEXUS_STATUS_ERR, "    Device Properties not found");
}

nxs_int DeviceImpl::createBuffer(size_t size, void *host_data) {
  APICALL(nxsCreateBuffer, id, size, 0, host_data);
}

nxs_int DeviceImpl::createCommandList() {
  if (auto fn = runtime->getFunction<nxsCreateCommandList_fn>(FN_nxsCreateCommandList)) {
    nxs_int bufId = (*fn)(id, 0);
    NEXUS_LOG(NEXUS_STATUS_NOTE, "  createCommandList " << bufId);
    if (bufId > -1)
      queues.push_back(bufId);
    return bufId;
  }
  return NXS_InvalidDevice;
}

#if 0
nxs_int Runtime::RTDevice::loadKernel() {
  
}

nxs_int Runtime::RTDevice::runKernel() {
  
}
#endif

///////////////////////////////////////////////////////////////////////////////
/// @return 
///////////////////////////////////////////////////////////////////////////////
Device::Device(detail::RuntimeImpl *rt, nxs_uint id) : Object(rt, id) {}

Device::Device() : Object() {}

void Device::release() const {
  get()->release();
}

Properties Device::getProperties() const { return get()->getProperties(); }

// Runtime functions
nxs_int Device::createCommandList() {
    return get()->createCommandList();
}

nxs_status Device::_copyBuffer(Buffer buf) {
  return get()->_copyBuffer(buf);
}