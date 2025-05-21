#ifndef NEXUS_LOG_H
#define NEXUS_LOG_H

#define NEXUS_STATUS_ERR " ERR"
#define NEXUS_STATUS_NOTE ""
#define NEXUS_STATUS_WARN " WARN"

#ifdef NEXUS_LOGGING
#include <iostream>

#define NEXUS_LOG(x, s) std::cerr << "[NEXUS][" << NEXUS_LOG_MODULE << "]" << x << ": " << s << std::endl

#else
#define NEXUS_LOG(x, s)
#endif

#endif // NEXUS_LOG_H