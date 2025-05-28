
#include <nexus/properties.h>
#include <nexus/log.h>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include <mutex>
#include <fstream>

using namespace nexus::detail;

#define NEXUS_LOG_MODULE "properties"

struct PropertiesImpl::PropMap {
  json _json;

  json getNode(const std::vector<std::string> &path) {
    json loc = _json;
    for (auto &key : path)
      loc = loc.at(key);
    return loc;
  }

  template <typename T>
  std::optional<T> getProp(const std::string &name) {
    try {
      return _json.at(name).get<T>();
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
};

PropertiesImpl::PropertiesImpl(const std::string &filepath)
: propertyFilePath(filepath), propertyMap(nullptr) {
}

void PropertiesImpl::loadProperties() {
  propertyMap = new PropMap;
  // Load json from file
  try {
    std::ifstream f(propertyFilePath);
    propertyMap->_json = json::parse(f);
    NEXUS_LOG(NEXUS_STATUS_NOTE, "loaded json from " << propertyFilePath << " - size: " << propertyMap->_json.size());
  } catch (...) {
    NEXUS_LOG(NEXUS_STATUS_ERR, "failed to load " << propertyFilePath);
  }
}


template <>
std::optional<std::string> PropertiesImpl::getProperty<std::string>(const std::string &propName) {
  std::call_once(loaded, [&]() { loadProperties(); });
  return propertyMap->getProp<std::string>(propName);
}

template <>
std::optional<int64_t> PropertiesImpl::getProperty<int64_t>(const std::string &propName) {
  std::call_once(loaded, [&]() { loadProperties(); });
  return propertyMap->getProp<int64_t>(propName);
}

template <>
std::optional<double> PropertiesImpl::getProperty<double>(const std::string &propName) {
  std::call_once(loaded, [&]() { loadProperties(); });
  return propertyMap->getProp<double>(propName);
}

template <>
std::optional<std::string> PropertiesImpl::getProperty<std::string>(const std::vector<std::string> &propPath) {
  std::call_once(loaded, [&]() { loadProperties(); });
  return propertyMap->getProp<std::string>(propPath);
}

template <>
std::optional<int64_t> PropertiesImpl::getProperty<int64_t>(const std::vector<std::string> &propPath) {
  std::call_once(loaded, [&]() { loadProperties(); });
  return propertyMap->getProp<int64_t>(propPath);
}
template <>
std::optional<double> PropertiesImpl::getProperty<double>(const std::vector<std::string> &propPath) {
  std::call_once(loaded, [&]() { loadProperties(); });
  return propertyMap->getProp<double>(propPath);
}
