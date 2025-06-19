#ifndef NEXUS_DEVICE_DB_H
#define NEXUS_DEVICE_DB_H

#include <nexus/properties.h>

#include <optional>
#include <string>
#include <unordered_map>

namespace nexus {

typedef std::unordered_map<std::string, Properties> DeviceMap;

const DeviceMap *getDeviceDB();

Properties lookupDevice(const std::string &archName);

}  // namespace nexus

#endif  // NEXUS_DEVICE_DB_H