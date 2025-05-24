// Created by alex on 5/24/25.
//

#ifndef COMMANDSCHEDULER_H
#define COMMANDSCHEDULER_H

#include "GraphInterface.h"
#include "Command.h"
#include <queue>


namespace Bcg {
    using CommandHandle = Vertex;
    using DependencyLink = Halfedge;

    class CommandScheduler {
    public:
        GraphOwning command_dag;
        VertexProperty<std::shared_ptr<Commands::AbstractCommand> > cmds;
        VertexProperty<int> in_degree;

        CommandScheduler() : command_dag(),
                             cmds(command_dag.add_vertex_property<std::shared_ptr<Commands::AbstractCommand> >(
                                 "cmd:ptrs")),
                             in_degree(command_dag.add_vertex_property<int>("cmd:in_degree", 0)) {
        }

        CommandHandle add_command(std::shared_ptr<Commands::AbstractCommand> cmd) {
            CommandHandle handle = command_dag.new_vertex();

            if (!command_dag.vertices.is_valid(handle)) {
                // Assuming Vertices container has is_valid
                Log::Error("Failed to create a new vertex for command.");
                return CommandHandle(); // Return an invalid handle
            }

            cmds[handle] = cmd;
            // in_degree[handle] will be 0 due to property default or explicit set above
            return handle;
        }

        // Add a dependency: prerequisite_cmd -> dependent_cmd
        bool add_dependency(CommandHandle prerequisite_cmd, CommandHandle dependent_cmd) {
            if (!command_dag.vertices.is_valid(prerequisite_cmd) ||
                !command_dag.vertices.is_valid(dependent_cmd)) {
                Log::Error("Invalid command handle for dependency.");
                return false;
            }

            if (prerequisite_cmd == dependent_cmd) {
                Log::Error("Cannot add self-dependency for command.");
                return false;
            }

            // Use new_edge to create the connection.
            // new_edge(v0, v1) returns the halfedge h01 (from v0 to v1).
            // Its opposite, h10, is also created.
            DependencyLink h_prereq_to_dep = command_dag.new_edge(prerequisite_cmd, dependent_cmd);

            if (!command_dag.halfedges.is_valid(h_prereq_to_dep)) {
                Log::Error("Failed to create dependency edge between commands.");
                return false;
            }

            // Explicitly set the direction using hdirection property
            // h_prereq_to_dep should be the "forward" direction
            command_dag.hdirection[h_prereq_to_dep] = true;
            // The opposite halfedge should be marked as not the primary direction (or false)
            DependencyLink h_opposite = command_dag.get_opposite(h_prereq_to_dep);
            if (command_dag.halfedges.is_valid(h_opposite)) {
                command_dag.hdirection[h_opposite] = false;
            }

            // Update in-degree of the dependent command
            in_degree[dependent_cmd]++;

            // Optional: Update vconnectivity if you want to use circulators based on it.
            // For a DAG, a vertex can have multiple outgoing dependencies.
            // vconnectivity stores only *one* outgoing halfedge.
            // If `command_dag.get_halfedge(prerequisite_cmd)` is invalid or you want to chain them,
            // you might set it. Or, you might not rely on vconnectivity for DAG traversal
            // if you iterate all halfedges.
            // For now, let's not modify vconnectivity here, as it's complex for multiple outgoing edges.

            return true;
        }

        void execute_commands() {
            std::vector<CommandHandle> sorted_commands_handles;
            std::queue<CommandHandle> q;

            // Initialize queue with nodes having in-degree 0
            for (CommandHandle h_cmd: command_dag.vertices) {
                // Assuming Vertices is iterable
                if (in_degree[h_cmd] == 0) {
                    q.push(h_cmd);
                }
            }

            int processed_count = 0;
            while (!q.empty()) {
                CommandHandle u_handle = q.front();
                q.pop();
                // sorted_commands_handles.push_back(u_handle); // Store handle if needed for later
                processed_count++;

                std::shared_ptr<Commands::AbstractCommand> cmd_to_execute = cmds[u_handle];
                if (cmd_to_execute) {
                    cmd_to_execute->execute();
                } else {
                    Log::Warn("No command found for vertex: {}", u_handle.idx());
                }

                // Iterate all outgoing halfedges to find successors of u_handle.
                for (DependencyLink h_dep: command_dag.get_halfedges(u_handle)) {
                    // Assuming HalfEdges is iterable
                    if (!command_dag.hdirection[h_dep]) {
                        // Only consider "forward" directed halfedges
                        continue;
                    }

                    CommandHandle v_dep = command_dag.get_vertex(h_dep); //this is the vertex h_dep points to
                    if (!command_dag.vertices.is_valid(v_dep)) {
                        Log::Error("Invalid dependent command handle: {}", v_dep.idx());
                        continue;
                    }
                    // Decrease in-degree of the dependent command
                    in_degree[v_dep]--;

                    if (in_degree[v_dep] == 0) {
                        // If in-degree becomes 0, add to queue for processing
                        q.push(v_dep);
                    }

                    if (in_degree[v_dep] < 0) {
                        // This should not happen in a well-formed DAG and means we have a cycle or mismanagement.
                        Log::Error("In-degree of command {} became negative: {} Probably a cycle or mismanagement", v_dep.idx(), in_degree[v_dep]);
                        return;
                    }
                }
            }

            if (processed_count != command_dag.vertices.size()) {
                // Assuming Vertices has num_elements()
                Log::Error("Cycle detected or unprocessed commands! Processed: {}, Expected: {}",
                           processed_count, command_dag.vertices.size());
                for (CommandHandle h_cmd: command_dag.vertices) {
                    if (in_degree[h_cmd] > 0) {
                        Log::Error("  - Command {} (Vertex {}) still has in_degree {}.",
                                   (cmds[h_cmd] ? cmds[h_cmd]->name : "UnknownCmd"),
                                   h_cmd.idx(),
                                   in_degree[h_cmd]);
                    }
                }
            }

            // Clear for next frame/batch
            clear_batch();
        }

        void clear_batch() {
            // How to clear the graph data?
            // GraphOwning owns `data`. GraphInterface references `data.vertices`, etc.
            // We need to clear the content of these containers.
            command_dag.vertices.clear(); // Assuming PropertyContainer has clear()
            command_dag.halfedges.clear();
            command_dag.edges.clear(); // If edges are used, otherwise not strictly needed for DAG

            // Properties are tied to the containers. When containers are cleared,
            // the property definitions might still exist but hold no data.
            // Re-getting them after a clear might be necessary if clear removes definitions.
            // Your PropertyContainer's `get_or_add` pattern is robust for this.
            cmds = command_dag.vertex_property<std::shared_ptr<Commands::AbstractCommand> >("cmd:ptrs");
            in_degree = command_dag.vertex_property<int>("cmd:in_degree", 0);
        }
    };
}

#endif //COMMANDSCHEDULER_H
