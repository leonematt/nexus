
#include <nexus/system.h>
#include <nexus/utility.h>
#include <nexus/log.h>

using namespace nexus;

#define NEXUS_LOG_MODULE "system"

/// @brief Construct a Platform for the current system
detail::SystemImpl::SystemImpl() {
  iterateEnvPaths("NEXUS_RUNTIME_PATH", "./runtime_libs", [&](const std::string &path, const std::string &name) {
    runtimes.emplace_back(path);
  });
}

/// @brief Get the System Platform
/// @return 
nexus::System nexus::getSystem() {
  static System s_system;
  return s_system;
}

