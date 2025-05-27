#ifndef NEXUS_DEVICE_H
#define NEXUS_DEVICE_H

//#include <nexus/property.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <optional>
#include <string>
#include <mutex>
#include <memory>

namespace nexus {

    class DeviceProperties {
        std::string devicePropertyFilePath;
        std::once_flag loaded;
        json propertyMap;
    public:
        DeviceProperties(const std::string &filepath)
            : devicePropertyFilePath(filepath) {}
        const json &getProperties() {
            std::call_once(loaded, [&]() { loadProperties(); });
            return propertyMap;
        }

    private:
        void loadProperties();
    };

    class Device {
        std::shared_ptr<DeviceProperties> deviceProps;
    public:
        template <typename... Args>
        Device(Args... args) : deviceProps(std::make_shared<DeviceProperties>(args...)) {}

        Device() = default;

        // Query Device Properties
        //   from name
        template <typename T>
        std::optional<const T> getProperty(const std::string &propName) const {
            auto props = deviceProps->getProperties();
            try {
                return props.at(propName).get<T>();
            } catch (...) {}
            return std::nullopt;
        }

        //   from path
        template <typename T>
        std::optional<T> getProperty(const std::vector<std::string> &propPath) const {
            auto loc = deviceProps->getProperties();
            for (auto &key : propPath) {
                try {
                    loc = loc.at(key);
                } catch (...) {
                    return std::nullopt;
                }
            }
            try {
                return loc.get<T>();
            } catch (...) {}
            return std::nullopt;
        }
    };

}

#endif // NEXUS_DEVICE_H
