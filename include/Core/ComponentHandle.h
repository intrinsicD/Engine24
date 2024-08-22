//
// Created by alex on 22.08.24.
//

#ifndef ENGINE24_COMPONENTHANDLE_H
#define ENGINE24_COMPONENTHANDLE_H

namespace Bcg{
    template<typename T>
    struct ComponentHandle {
        T *component = nullptr;
        bool valid = false;

        ComponentHandle() = default;

        ComponentHandle(T *component) : component(component), valid(true) {}

        T &operator*() {
            return *component;
        }

        T *operator->() {
            return component;
        }

        operator bool() const {
            return valid;
        }
    };
}

#endif //ENGINE24_COMPONENTHANDLE_H
