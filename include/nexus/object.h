#ifndef NEXUS_OBJECT_H
#define NEXUS_OBJECT_H

#include <memory>
#include <nexus-api.h>

namespace nexus {

    namespace detail {

        // All Actual objects need an owner (except System)
        // + and ID within the owner
        template <typename Towner>
        class OwnerRef {
            // TODO: use CRTP for Impl?
        public:

            OwnerRef(Towner *_owner, nxs_int _id)
                : owner(_owner), id(_id) {}

            Towner *getOwner() const { return owner; }
            nxs_int getId() const { return id; }
        private:
            Towner *owner;
            nxs_int id;
        };
    }

    // Facade base-class
    template <typename Timpl>
    class Object {
        // set of runtimes
        typedef std::shared_ptr<Timpl> ImplRef;
        ImplRef impl;
    
    public:
        template <typename... Args>
        Object(Args... args) : impl(std::make_shared<Timpl>(args...)) {}

        // Empty CTor - assumes Impl doesn't have an empty CTOR
        Object() = default;

        operator bool() const { return impl; }
        ImplRef get() const { return impl; }
    };
    
}

#endif // NEXUS_OBJECT_H
