
#include <nexus/properties.h>
#include <nexus/log.h>

#include <fstream>

using namespace nexus;

#define NEXUS_LOG_MODULE "device"

void PropertyCache::loadProperties() {
  // Load json from file
  try {
    std::ifstream f(propertyFilePath);
    propertyMap = json::parse(f);
    NEXUS_LOG(NEXUS_STATUS_NOTE, "loaded json from " << propertyFilePath << " - size: " << propertyMap.size());
  } catch (...) {
    NEXUS_LOG(NEXUS_STATUS_ERR, "failed to load " << propertyFilePath);
  }
}
