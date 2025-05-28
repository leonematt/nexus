#ifndef NEXUS_OBJECT_H
#define NEXUS_OBJECT_H

#include <memory>

namespace nexus {

    // Facade base-class
    template <typename T>
    class Object {
        // set of runtimes
        typedef T Impl;
        std::shared_ptr<Impl> impl;
    
    public:
        template <typename... Args>
        Object(Args... args) : impl(std::make_shared<T>(args...)) {}

        // Empty CTor - assumes Impl doesn't have an empty CTOR
        Object() = default;

        operator bool() const { return impl; }
        std::shared_ptr<Impl> get() const { return impl; }
    };
    
}

#endif // NEXUS_OBJECT_H
