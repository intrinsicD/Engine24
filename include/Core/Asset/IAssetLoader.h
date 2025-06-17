//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_IASSETLOADER_H
#define ENGINE24_IASSETLOADER_H

#include <memory>
#include "IAsset.h"

namespace Bcg {
    class IAssetLoader {
    public:
        virtual ~IAssetLoader() = default;

        // Loads asset data from a file path into a CPU-side IAsset object.
        virtual std::shared_ptr<IAsset> load_asset(const std::string &path) = 0;

        // A list of file extensions this loader supports (e.g., ".png", ".jpg").
        virtual const std::vector<std::string> &get_supported_extensions() const = 0;
    };
}

#endif //ENGINE24_IASSETLOADER_H
