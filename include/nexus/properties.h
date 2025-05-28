#ifndef NEXUS_PROPERTIES_H
#define NEXUS_PROPERTIES_H

#include <nexus-api.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <optional>
#include <string>
#include <mutex>
#include <memory>

namespace nexus {

    class PropertyCache {
        std::string propertyFilePath;
        std::once_flag loaded;
        json propertyMap;
    public:
        PropertyCache(const std::string &filepath)
            : propertyFilePath(filepath) {}
        const json &getProperties() {
            std::call_once(loaded, [&]() { loadProperties(); });
            return propertyMap;
        }

    private:
        void loadProperties();
    };

    class Properties {
        std::shared_ptr<PropertyCache> properties;
    public:
        template <typename... Args>
        Properties(Args... args) : properties(std::make_shared<PropertyCache>(args...)) {}

        Properties() = default;

        // Query Device Properties
        //   from name
        template <typename T>
        std::optional<const T> getProperty(const std::string &propName) const {
            auto props = properties->getProperties();
            try {
                return props.at(propName).get<T>();
            } catch (...) {}
            return std::nullopt;
        }

        //   from path
        template <typename T>
        std::optional<T> getProperty(const std::vector<std::string> &propPath) const {
            auto loc = properties->getProperties();
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
        //   from name
        template <typename T>
        std::optional<const T> getProperty(NXSAPI_PropertyEnum prop) const {
            auto props = properties->getProperties();
            const char *propName = nxsGetPropName(prop);
            try {
                return props.at(propName).get<T>();
            } catch (...) {}
            return std::nullopt;
        }

        //   from path
        template <typename T>
        std::optional<T> getProperty(const std::vector<NXSAPI_PropertyEnum> &propPath) const {
            auto loc = properties->getProperties();
            for (auto key : propPath) {
                const char *keyStr = nxsGetPropName(key);
                try {
                    loc = loc.at(keyStr);
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

#endif // NEXUS_PROPERTIES_H
