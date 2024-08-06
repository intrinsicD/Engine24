//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_KDTREENODE_H
#define ENGINE24_KDTREENODE_H

namespace Bcg{
    struct KDNode {
        float split_value;
        int left, right, index;
    };
}

#endif //ENGINE24_KDTREENODE_H
