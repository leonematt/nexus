#ifndef NEXUS_DATABASE_H
#define NEXUS_DATABASE_H

#include <nexus/device.h>

#include <unordered_map>
#include <optional>
#include <string>

namespace nexus {

    typedef std::unordered_map<std::string, Device> DeviceMap;

    std::optional<Device> lookupDevice(const std::string &archName);

}

#endif // NEXUS_DATABASE_H