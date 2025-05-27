#ifndef NEXUS_PROPERTY_H
#define NEXUS_PROPERTY_H

#include <variant>
#include <string>
#include <vector>

/// @brief  NOT USED, see json in Device or nexus-api
namespace nexus {

  using Prop = std::variant<int64_t, double, std::string>; // , std::vector<Property>>;

  using PropVec = std::vector<Prop>;

  using Property = std::variant<Prop, PropVec>;

  template <typename T>
  T get(Property p) {
    return std::get<T>(std::get<Prop>(p));
  }

}

#endif // NEXUS_PROPERTY_H
