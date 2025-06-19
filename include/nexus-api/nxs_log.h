#ifndef NEXUS_API_LOG_H
#define NEXUS_API_LOG_H

#define NXSAPI_STATUS_ERR " ERROR"
#define NXSAPI_STATUS_NOTE ""
#define NXSAPI_STATUS_WARN " WARN"

#if defined(NXSAPI_LOGGING) && defined(__cplusplus)
#include <iomanip>
#include <iostream>

#define NXSAPI_LOG(STATUS, s)                                                 \
  {                                                                           \
    const char *_log_prefix =                                                 \
        "[NEXUS-API][" NXSAPI_LOG_MODULE "]" STATUS ": ";                     \
    std::cerr << std::left << std::setw(50) << _log_prefix << s << std::endl; \
  }

#else
#define NXSAPI_LOG(x, s)
#endif

#endif  // NEXUS_API_LOG_H