#ifndef NEXUS_PROPERTY_H
#define NEXUS_PROPERTY_H

#include <variant>
#include <string>
#include <vector>

#include <nexus-api.h>

/// @brief  NOT USED, see json in Device or nexus-api
namespace nexus {

  using Prop = std::variant<nxs_long, nxs_double, std::string>;
  using PropIntVec = std::vector<nxs_long>;
  using PropFltVec = std::vector<nxs_double>;
  using PropStrVec = std::vector<std::string>;

  using PropVariant = std::variant<Prop, PropStrVec, PropIntVec, PropFltVec>;

  class Property : public PropVariant {
  public:
    using PropVariant::PropVariant;

    template <nxs_property Tnp>
    typename nxsPropertyType<Tnp>::type getValue() const {
      return std::get<typename nxsPropertyType<Tnp>::type>(std::get<Prop>(*this));
    }

    template <typename T>
    T getValue() const {
      return std::get<T>(std::get<Prop>(*this));
    }

    template <typename T>
    std::vector<T> getValueVec() const {
      return std::get<std::vector<T>>(*this);
    }
  };

}

#endif // NEXUS_PROPERTY_H
