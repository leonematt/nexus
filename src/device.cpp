
#include <nexus/device.h>
#include <nexus/log.h>

#include <fstream>

using namespace nexus;

#define NEXUS_LOG_MODULE "device"

void DeviceProperties::loadProperties() {
  // Load json from file
  try {
    std::ifstream f(devicePropertyFilePath);
    propertyMap = json::parse(f);
    NEXUS_LOG(NEXUS_STATUS_NOTE, "loaded json from " << devicePropertyFilePath << " - size: " << propertyMap.size());
  } catch (...) {
    NEXUS_LOG(NEXUS_STATUS_ERR, "failed to load " << devicePropertyFilePath);
  }
}
