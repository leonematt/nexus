
#include <nexus/properties.h>
#include <nexus/log.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <mutex>
#include <fstream>

using namespace nexus;
using namespace nexus::detail;

#define NEXUS_LOG_MODULE "properties"

namespace nexus {
namespace detail {
  class PropertiesImpl {
    std::string propertyFilePath;
    std::once_flag loaded;
    json props;
  public:
    PropertiesImpl(const std::string &filepath) : propertyFilePath(filepath) {}

    template <typename T>
    std::optional<T> getProperty(const std::string &propName) {
      std::call_once(loaded, [&]() { loadProperties(); });
      return getProp<T>(propName);
    }

    template <typename T>
    std::optional<T> getProperty(const std::vector<std::string> &propPath) {
      std::call_once(loaded, [&]() { loadProperties(); });
      return getProp<T>(propPath);
    }

  private:
    json getNode(const std::vector<std::string> &path) {
      json loc = props;
      for (auto &key : path)
        loc = loc.at(key);
      return loc;
    }

    template <typename T>
    std::optional<T> getProp(const std::string &name) {
      try {
        return props.at(name).get<T>();
      } catch (...) {}
      return std::nullopt;
    }

    template <typename T>
    std::optional<T> getProp(const std::vector<std::string> &path) {
      try {
        return getNode(path).get<T>();
      } catch (...) {}
      return std::nullopt;
    }

  private:
    void loadProperties() {
      // Load json from file
      try {
        std::ifstream f(propertyFilePath);
        props = json::parse(f);
        NEXUS_LOG(NEXUS_STATUS_NOTE, "loaded json from " << propertyFilePath << " - size: " << props.size());
      } catch (...) {
        NEXUS_LOG(NEXUS_STATUS_ERR, "failed to load " << propertyFilePath);
      }
    }
  };

}
}


///////////////////////////////////////////////////////////////////////////////
/// @brief
///////////////////////////////////////////////////////////////////////////////
Properties::Properties(const std::string &filepath) : Object(filepath) {}

template <>
std::optional<std::string> Properties::getProperty<std::string>(const std::string &name) const {
  return get()->getProperty<std::string>(name);
}

template <>
std::optional<int64_t> Properties::getProperty<int64_t>(const std::string &name) const {
  return get()->getProperty<int64_t>(name);
}

template <>
std::optional<double> Properties::getProperty<double>(const std::string &name) const {
  return get()->getProperty<double>(name);
}

template <>
std::optional<std::string> Properties::getProperty<std::string>(const std::vector<std::string> &path) const {
  return get()->getProperty<std::string>(path);
}

template <>
std::optional<int64_t> Properties::getProperty<int64_t>(const std::vector<std::string> &path) const {
  return get()->getProperty<int64_t>(path);
}
template <>
std::optional<double> Properties::getProperty<double>(const std::vector<std::string> &path) const {
  return get()->getProperty<double>(path);
}


