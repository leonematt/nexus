#ifndef NEXUS_LIBRARY_H
#define NEXUS_LIBRARY_H

#include <nexus/object.h>
#include <nexus/kernel.h>
#include <nexus-api.h>

#include <list>

namespace nexus {
    
    namespace detail {
        class DeviceImpl; // owner
        class LibraryImpl;
    }

    // System class
    class Library : public Object<detail::LibraryImpl, detail::DeviceImpl> {
        friend OwnerTy;
    public:
        Library(detail::Impl owner);
        using Object::Object;

        void release() const;
        nxs_int getId() const override;

        std::optional<Property> getProperty(nxs_int prop) const override;

        Kernel getKernel(const std::string &kernelName);
    };

    typedef Objects<Library> Librarys;

}

#endif // NEXUS_LIBRARY_H