//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_IASSET_H
#define ENGINE24_IASSET_H

#include "AssetHandle.h"

namespace Bcg{
    class IAsset {
    public:
        virtual ~IAsset() = default;

        AssetHandle handle;
    };
}

#endif //ENGINE24_IASSET_H
