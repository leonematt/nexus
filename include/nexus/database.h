#ifndef NEXUS_DATABASE_H
#define NEXUS_DATABASE_H

//#include <nexus/property.h>

#include <optional>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <unordered_map>
#include <optional>

namespace nexus {

    class DevicePropMap {
        std::string fileName;
        json propertyMap;
    public:
        DevicePropMap(const char *filename);

        template <typename T>
        std::optional<T> getProperty(const std::string &propName) const {
            try {
                return propertyMap.at(propName).get<T>();
            } catch (...) {}
            return std::nullopt;
        }
#if 0
        template <typename T>
        std::optional<T> getProperty(const std::vector<std::string> &propPath) const {

        }
        #endif
    };

    typedef std::unordered_map<std::string, DevicePropMap> DeviceMap;

    std::optional<const DevicePropMap> lookupDevice(const char *archName);

}

#endif // NEXUS_DATABASE_H