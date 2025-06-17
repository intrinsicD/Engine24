//
// Created by alex on 17.06.25.
//

#ifndef ENGINE24_ASSETHANDLE_H
#define ENGINE24_ASSETHANDLE_H

#include <cstdint>
#include <functional>

namespace Bcg {
    using AssetID = uint32_t;

    class AssetHandle {
    public:
        AssetHandle() : m_id(0) {}

        explicit AssetHandle(AssetID id) : m_id(id) {}

        AssetID get_id() const { return m_id; }

        bool is_valid() const { return m_id != 0; }

        bool operator==(const AssetHandle &other) const { return m_id == other.m_id; }

        bool operator!=(const AssetHandle &other) const { return m_id != other.m_id; }

    private:
        AssetID m_id;
    };
}

namespace std {
    template<>
    struct hash<Bcg::AssetHandle> {
        size_t operator()(const Bcg::AssetHandle& handle) const {
            return hash<Bcg::AssetID>()(handle.get_id());
        }
    };
}

#endif //ENGINE24_ASSETHANDLE_H
