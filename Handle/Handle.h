//
// Created by alex on 23.07.24.
//

#ifndef ENGINE24_HANDLE_H
#define ENGINE24_HANDLE_H

namespace Bcg {
#define BCG_INVALID_HANDLE -1

    template<typename T>
    struct Handle {
        unsigned int id = BCG_INVALID_HANDLE;

        operator bool() const {
            return id != BCG_INVALID_HANDLE;
        }
    };
}

#endif //ENGINE24_HANDLE_H
