#ifndef NEXUS_UTILITY_H
#define NEXUS_UTILITY_H

#include <functional>
#include <string>
#include <vector>

namespace nexus {

typedef std::function<void(const std::string &, const std::string &)>
    PathNameFn;
void iterateEnvPaths(const char *envVar, const char *envDefault,
                     const PathNameFn &func);

}  // namespace nexus

#endif  // NEXUS_SYSTEM_H