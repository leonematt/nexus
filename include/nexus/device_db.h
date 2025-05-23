#ifndef NEXUS_DEVICE_DB_H
#define NEXUS_DEVICE_DB_H

#include <nexus/device.h>

#include <unordered_map>
#include <optional>
#include <string>

namespace nexus {

    typedef std::unordered_map<std::string, Device> DeviceMap;

    std::optional<Device> lookupDevice(const std::string &archName);

}

#endif // NEXUS_DEVICE_DB_H