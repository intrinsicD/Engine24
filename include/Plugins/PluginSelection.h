//
// Created by alex on 06.08.24.
//

#ifndef ENGINE24_PLUGINSELECTION_H
#define ENGINE24_PLUGINSELECTION_H

#include "Plugin.h"
#include "Command.h"
#include "MatVec.h"
#include <set>

namespace Bcg {
    enum class SelectionMode {
        None,
        Entity,
        Vertex,
        Edge,
        Face,
    };

    struct SelectedEntities {
        std::set<entt::entity> entities;
    };

    struct SelectedVertices {
        std::set<size_t> vertices;
    };

    struct SelectedEdges {
        std::set<size_t> edges;
    };

    struct SelectedFaces {
        std::set<size_t> faces;
    };

    class PluginSelection : public Plugin {
    public:
        PluginSelection();

        ~PluginSelection() = default;

        void activate() override;

        void begin_frame() override;

        void update() override;

        void end_frame() override;

        void deactivate() override;

        void render_menu() override;

        void render_gui() override;

        void render() override;

        static SelectionMode get_current_selection_mode();

        static void set_current_selection_mode(SelectionMode mode);

        static void disable_selection();

        static SelectedEntities &get_selected_entities();

        static SelectedVertices &get_selected_vertices(entt::entity entity_id);

        static void mark_selected_vertices(entt::entity entity_id, const Vector<float, 3> &color);

        static SelectedEdges &get_selected_edges(entt::entity entity_id);

        static void mark_selected_edges(entt::entity entity_id, const Vector<float, 3> &color);

        static SelectedFaces &get_selected_faces(entt::entity entity_id);

        static void mark_selected_faces(entt::entity entity_id, const Vector<float, 3> &color);
    };

    namespace Commands {
        struct MarkPoints : public AbstractCommand {
            MarkPoints(entt::entity entity_id, const std::string &property_name) : AbstractCommand("MarkPoints"),
                entity_id(entity_id),
                property_name(property_name) {
            }

            void execute() const override;

            entt::entity entity_id;
            std::string property_name;
        };

        struct EnableVertexSelection : public AbstractCommand {
            EnableVertexSelection(entt::entity entity_id, const std::string &property_name) : AbstractCommand(
                    "EnableVertexSelection"),
                entity_id(entity_id) {
            }

            void execute() const override;

            entt::entity entity_id;
        };
    }
}

#endif //ENGINE24_PLUGINSELECTION_H
