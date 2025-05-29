#ifndef NEXUS_LOG_H
#define NEXUS_LOG_H

#define NEXUS_STATUS_ERR " ERROR"
#define NEXUS_STATUS_NOTE ""
#define NEXUS_STATUS_WARN " WARN"

#ifdef NEXUS_LOGGING
#include <iostream>
#include <iomanip>

#define NEXUS_LOG(STATUS, s)  {\
    const char *_log_prefix = "[NEXUS][" NEXUS_LOG_MODULE "]" STATUS ": "; \
    std::cerr << std::left << std::setw(30) << _log_prefix << s << std::endl; }

#else
#define NEXUS_LOG(x, s)
#endif

#endif // NEXUS_LOG_H