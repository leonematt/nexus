#ifndef NEXUS_PROPERTIES_H
#define NEXUS_PROPERTIES_H

#include <nexus/object.h>
#include <nexus-api.h>

#include <optional>
#include <string>
#include <vector>
#include <mutex>
#include <algorithm>

namespace nexus {

    namespace detail {
        class PropertiesImpl {
            std::string propertyFilePath;
            std::once_flag loaded;
            struct PropMap;
            PropMap *propertyMap;
        public:
            PropertiesImpl(const std::string &filepath);

            template <typename T>
            std::optional<T> getProperty(const std::string &propName);

            template <typename T>
            std::optional<T> getProperty(const std::vector<std::string> &propPath);

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
        std::optional<T> getProperty(const std::string &propName) const {
            return get()->getProperty<T>(propName);
        }

        //   from path
        template <typename T>
        std::optional<T> getProperty(const std::vector<std::string> &propPath) const {
            return get()->getProperty<T>(propPath);
        }

        //   from name
        template <typename T>
        std::optional<T> getProperty(NXSAPI_PropertyEnum prop) const {
            return getProperty<T>(nxsGetPropName(prop));
        }

        //   from path
        template <typename T>
        std::optional<T> getProperty(const std::vector<NXSAPI_PropertyEnum> &propPath) const {
            std::vector<std::string> propNames;
            std::for_each(propPath.begin(), propPath.end(),
                 [&](NXSAPI_PropertyEnum pn) { propNames.push_back(nxsGetPropName(pn)); });
            return getProperty<T>(propNames);
        }
    };

}

#endif // NEXUS_PROPERTIES_H
