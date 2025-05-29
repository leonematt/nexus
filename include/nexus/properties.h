#ifndef NEXUS_PROPERTIES_H
#define NEXUS_PROPERTIES_H

#include <nexus/object.h>
#include <nexus-api.h>

#include <optional>
#include <string>
#include <vector>
#include <algorithm>

namespace nexus {

    namespace detail {
        class PropertiesImpl;
    }
    class Properties : Object<detail::PropertiesImpl> {
    public:
        Properties(const std::string &filepath);
        using Object::Object;

        // Query Device Properties
        //   from name
        template <typename T>
        std::optional<T> getProperty(const std::string &name) const;
        //   from path
        template <typename T>
        std::optional<T> getProperty(const std::vector<std::string> &path) const;

        //   from name
        template <typename T>
        std::optional<T> getProperty(nxs_property prop) const {
            return getProperty<T>(nxsGetPropName(prop));
        }

        //   from path
        template <typename T>
        std::optional<T> getProperty(const std::vector<nxs_property> &propPath) const {
            std::vector<std::string> names;
            std::for_each(propPath.begin(), propPath.end(),
                 [&](nxs_property pn) { names.push_back(nxsGetPropName(pn)); });
            return getProperty<T>(names);
        }
    };

}

#endif // NEXUS_PROPERTIES_H
