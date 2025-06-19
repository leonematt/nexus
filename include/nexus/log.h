#ifndef NEXUS_LOG_H
#define NEXUS_LOG_H

#define NEXUS_STATUS_ERR " ERROR"
#define NEXUS_STATUS_NOTE ""
#define NEXUS_STATUS_WARN " WARN"

#define NEXUS_LOG_DEPTH 30

#ifdef NEXUS_LOGGING
#include <iomanip>
#include <iostream>

#define NEXUS_LOG(STATUS, s)                           \
  std::cerr << std::left << std::setw(NEXUS_LOG_DEPTH) \
            << "[NEXUS][" NEXUS_LOG_MODULE "]" STATUS ": " << s << std::endl

#else
#define NEXUS_LOG(x, s)
#endif

#endif  // NEXUS_LOG_H