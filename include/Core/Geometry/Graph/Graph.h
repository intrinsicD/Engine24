//
// Created by alex on 26.08.24.
//

#ifndef ENGINE24_GRAPH_H
#define ENGINE24_GRAPH_H

#include "GraphInterface.h"

namespace Bcg {
    class Graph {
    public:
        Graph() : interface(data) {
            ;
        }

        virtual ~Graph() = default;

        Graph(const Graph &other) : data(other.data),
                             interface(data) {

        }

        Graph &operator=(const Graph &other) {
            if (this != &other) {
                data = other.data;
                interface = GraphInterface(data);
            }
            return *this;
        }

        // Define move constructor
        Graph(Graph &&other) noexcept
            : data(std::move(other.data)),
              interface(data) {

        }

        // Define move assignment operator
        Graph &operator=(Graph &&other) noexcept {
            if (this != &other) {
                data = std::move(other.data);
                interface = GraphInterface(data);
            }
            return *this;
        }

        GraphData data;
        GraphInterface interface;
    };
}

#endif //ENGINE24_GRAPH_H
