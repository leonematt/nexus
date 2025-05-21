#include <nexus.h>

#include <magic_enum/magic_enum.hpp>

std::string nexus::getPropName(nexus::DeviceProp p) {
  return magic_enum::enum_name(p).data();
}

