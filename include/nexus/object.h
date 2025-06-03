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
        public:

            typedef Towner OwnerTy;

            OwnerRef(Towner *_owner, nxs_int _id)
                : owner(_owner), id(_id) {}

            nxs_int getId() const { return id; }

        protected:
            // Only the derived class can access
            Towner *getOwner() const { return owner; }

        private:
            Towner *owner;
            nxs_int id;
        };
    }

    // Facade base-class
    template <typename Timpl, typename Towner = void>
    class Object {
        // set of runtimes
        typedef std::shared_ptr<Timpl> ImplRef;
        ImplRef impl;
    
    public:

        typedef Towner OwnerTy;
        typedef detail::OwnerRef<Towner> OwnerRef;
        
        template <typename... Args>
        Object(OwnerRef owner, Args... args) : impl(std::make_shared<Timpl>(owner, args...)) {}

        template <typename... Args>
        Object(Args... args) : impl(std::make_shared<Timpl>(args...)) {}

        // Empty CTor - assumes Impl doesn't have an empty CTOR
        Object() = default;
        virtual ~Object() {}

        operator bool() const { return impl; }
        bool operator ==(const Object &that) const { return impl == that.impl; }

        virtual void release() { impl = nullptr; }

        virtual nxs_int getId() const = 0;

    protected:
        ImplRef get() const { return impl; }
    };
    
}

#endif // NEXUS_OBJECT_H
