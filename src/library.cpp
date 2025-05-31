
#include <nexus/library.h>
#include <nexus/log.h>

#include "_device_impl.h"

#define NEXUS_LOG_MODULE "library"

using namespace nexus;
using namespace nexus::detail;

namespace nexus {
namespace detail {
  class LibraryImpl {
  public:
    /// @brief Construct a Platform for the current system
    LibraryImpl(DeviceImpl *_dev, nxs_int _id)
      : device(_dev), id(_id) {
        NEXUS_LOG(NEXUS_STATUS_NOTE, "  Library: " << id);
      }

    ~LibraryImpl() {
      NEXUS_LOG(NEXUS_STATUS_NOTE, "  ~Library: " << id);
      release();
    }

    void release() {
      device->releaseLibrary(id);
    }

  private:
    DeviceImpl *device;
    nxs_int id;

    // set of runtimes
    size_t size;
    void *data;
  };
}
}


///////////////////////////////////////////////////////////////////////////////
Library::Library(detail::DeviceImpl *_dev, nxs_int _id)
  : Object(_dev, _id) {}

Library::Library() : Object() {}

void Library::release() const {
  get()->release();
}

