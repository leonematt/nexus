#ifndef NEXUS_OBJECT_H
#define NEXUS_OBJECT_H

#include <memory>
#include <nexus-api.h>

namespace nexus {

    namespace detail {

        // All Actual objects need an owner (except System)
        // + and ID within the owner
        class Impl {
        public:

            Impl(Impl *_owner = nullptr, nxs_int _id = 0)
                : owner(_owner), id(_id) {}
            virtual ~Impl() {}

            nxs_int getId() const { return id; }

        protected:
            // Only the derived class can access
            template <typename T = Impl>
            T *getParent() const { return dynamic_cast<T*>(owner); }

            template <typename T>
            T *getParentOfType() const {
                if (auto *par = dynamic_cast<T*>(owner))
                    return par;
                if (owner)
                    return owner->getParentOfType<T>();
                return nullptr;
            }

            void setId(nxs_int nid) { id = nid; }

        private:
            Impl *owner;
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
        
        template <typename... Args>
        Object(detail::Impl owner, Args... args) : impl(std::make_shared<Timpl>(owner, args...)) {}

        template <typename... Args>
        Object(Args... args) : impl(std::make_shared<Timpl>(args...)) {}

        // Empty CTor - assumes Impl doesn't have an empty CTOR
        Object() = default;
        virtual ~Object() {}

        operator bool() const { return impl && getId() >= 0; }
        bool operator ==(const Object &that) const { return impl == that.impl; }

        virtual void release() { impl = nullptr; }

        virtual nxs_int getId() const = 0;

    protected:
        ImplRef get() const { return impl; }
    };
    
}

#endif // NEXUS_OBJECT_H
