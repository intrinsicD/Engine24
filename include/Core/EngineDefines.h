//
// Created by alex on 10.09.24.
//

#ifndef ENGINE24_ENGINEDEFINES_H
#define ENGINE24_ENGINEDEFINES_H

namespace Bcg{
    #define BCG_VERSION_MAJOR 0
    #define BCG_VERSION_MINOR 1
    #define BCG_VERSION_PATCH 0

    #define BCG_VERSION (BCG_VERSION_MAJOR * 10000 + BCG_VERSION_MINOR * 100 + BCG_VERSION_PATCH)

    #define BCG_VERSION_STRING "0.1.0"

    #define BCG_NAMESPACE Bcg
    #define BCG_INLINE inline

    #define BCG_NAMESPACE_BEGIN namespace BCG_NAMESPACE{
    #define BCG_NAMESPACE_END }

    #define BCG_USING_NAMESPACE using namespace BCG_NAMESPACE;

    #define BCG_USING_NAMESPACE_BEGIN using namespace BCG_NAMESPACE{
    #define BCG_USING_NAMESPACE_END
}

#endif //ENGINE24_ENGINEDEFINES_H
