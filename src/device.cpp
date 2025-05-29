#include <nexus/device_db.h>
#include <nexus/runtime.h>
#include <nexus/log.h>

#define NEXUS_LOG_MODULE "device"

using namespace nexus;
using namespace nexus::detail;

namespace nexus {
namespace detail {
// RTDevice - wrapper for Device properties and Runtime actions
class DeviceImpl {
    RuntimeImpl *runtime;
    nxs_uint id;
    Properties deviceProps;
    std::vector<nxs_uint> buffers;
    std::vector<nxs_uint> queues;
public:
    DeviceImpl(RuntimeImpl *rt, nxs_uint id);
    ~DeviceImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "    ~Device: " << id);
    }

    Properties getProperties() const { return deviceProps; }

    // Runtime functions
    nxs_int createBuffer(size_t size, void *hostData = nullptr);
    nxs_int createCommandList();

};

}
}


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
  if (auto fn = runtime->getFunction<nxsCreateBuffer_fn>(FN_nxsCreateBuffer)) {
    nxs_int bufId = (*fn)(id, size, 0, host_data);
    if (bufId > -1)
      buffers.push_back(bufId);
    return bufId;
  }
  return NXS_InvalidDevice;
}

nxs_int DeviceImpl::createCommandList() {
  if (auto fn = runtime->getFunction<nxsCreateCommandList_fn>(FN_nxsCreateCommandList)) {
    nxs_int bufId = (*fn)(id, 0);
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

Properties Device::getProperties() const { return get()->getProperties(); }

// Runtime functions
nxs_int Device::createBuffer(size_t size, void *hostData) {
    return get()->createBuffer(size, hostData);
}
nxs_int Device::createCommandList() {
    return get()->createCommandList();
}
