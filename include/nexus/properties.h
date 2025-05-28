#ifndef NEXUS_PROPERTIES_H
#define NEXUS_PROPERTIES_H

#include <nexus/object.h>
#include <nexus-api.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <optional>
#include <string>
#include <mutex>
#include <memory>

namespace nexus {

    namespace detail {
        class PropertiesImpl {
            std::string propertyFilePath;
            std::once_flag loaded;
            json propertyMap;
        public:
            PropertiesImpl(const std::string &filepath)
                : propertyFilePath(filepath) {}
            
            const json &getProperties() {
                std::call_once(loaded, [&]() { loadProperties(); });
                return propertyMap;
            }

        private:
            void loadProperties();
        };
    }

    class Properties : Object<detail::PropertiesImpl> {
    public:
        using Object::Object;

        // Query Device Properties
        //   from name
        template <typename T>
        std::optional<const T> getProperty(const std::string &propName) const {
            auto props = get()->getProperties();
            try {
                return props.at(propName).get<T>();
            } catch (...) {}
            return std::nullopt;
        }

        //   from path
        template <typename T>
        std::optional<T> getProperty(const std::vector<std::string> &propPath) const {
            auto loc = get()->getProperties();
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
            auto props = get()->getProperties();
            const char *propName = nxsGetPropName(prop);
            try {
                return props.at(propName).get<T>();
            } catch (...) {}
            return std::nullopt;
        }

        //   from path
        template <typename T>
        std::optional<T> getProperty(const std::vector<NXSAPI_PropertyEnum> &propPath) const {
            auto loc = get()->getProperties();
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
