#ifndef NEXUS_PROPERTY_H
#define NEXUS_PROPERTY_H

#include <variant>
#include <string>
#include <vector>

#include <nexus-api.h>

/// @brief  NOT USED, see json in Device or nexus-api
namespace nexus {

  using Prop = std::variant<nxs_long, nxs_double, std::string>;

  using PropVec = std::vector<Prop>;

  using Property = std::variant<Prop, PropVec>;

  template <nxs_property Tnp>
  typename nxsPropertyType<Tnp>::type getPropertyValue(Property p) {
    return std::get<typename nxsPropertyType<Tnp>::type>(std::get<Prop>(p));
  }

  template <typename T>
  T getPropertyValue(Property p) {
    return std::get<T>(std::get<Prop>(p));
  }

}

#endif // NEXUS_PROPERTY_H
