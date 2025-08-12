//
// Created by alex on 12.08.25.
//

#ifndef ENGINE24_STRINGTRAITSPOINTCLOUD_H
#define ENGINE24_STRINGTRAITSPOINTCLOUD_H

#include "StringTraits.h"
#include "PointCloud.h"

namespace Bcg{
    template<>
    struct StringTraits<PointCloud> {
        static std::string ToString(const PointCloud &t) {
            return "PointCloud to string not jet implemented";
        }
    };
}

#endif //ENGINE24_STRINGTRAITSPOINTCLOUD_H
