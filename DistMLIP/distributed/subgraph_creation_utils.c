#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include "subgraph_creation_utils.h"
#include <stdlib.h>

const double EPSILON = 1e-10; // tiny value used to move walls in order to make sure atoms don't collide with walls
const long DEFAULT_ITEMS = 20; // the default # of UDE/DE to initialize the edge arrays for each entry in the adj_list table, also the default # of UDE in PartitionTransferInfo
const long LINE_GRAPH_DEFAULT_ITEMS = 1000000; // TODO: ultimately, we would want this to scale with the input, but we'll keep it constant for now...
                                                // TODO: also, for reallocating, it will probably be faster to have minimal reallocs (so exponential growth within memory increase)
                                                // this might require the use of some variable to keep track of the size of the memory array
const double NUMERICAL_TOL = 1e-8; // numerical tolerance for determining how close two doubles are to each other


// array of size num_nodes where the ith index of to_partition is the partition in which
// node i is sent to. if node i is not a border node, then to_partition[i] will be -1
int* nodes_to_partition;

/*
Creates structs with necessary node features + metadata + edge information

WARNING NOTE: partitions created are naive vertical slices
*/
Results* get_features(
    long num_edges,
    long* src_nodes,
    long* dst_nodes,
    double* center_coords, // cartesian coordinates of each node: (num_nodes, 3)
    long num_nodes,
    unsigned int num_partitions,
    double atom_cutoff,
    double* distances, // shape: (num_edges, )
    double bond_cutoff,
    long num_within_bond_r_indices,
    long* within_bond_r_indices,
    double* images, // shape: (num_edges, 3), c-contiguous (contiguous in the 3-dimension) [:, ::1]
    int num_threads, // number of threads to use for parallelized regions
    bool use_bond_graph, // whether or not to calculate the bond graph
    double* frac_coords, // fractional coordinates of each node: (num_nodes, 3) TODO: LEFT OFF HERE (1.19.2025 7:48 PM) working on integrating fractional coordinates to calculate vertical wall partition dimension
    double* lattice // lattice matrix (3x3), c-contiguous, [:, ::1]
) {
    #ifdef TIMING
        struct timespec t0 = get_time();
    #endif
    if (num_partitions == 1) {
        printf("Only 1 partition. That's not gonna work... bruh.\n");
        return NULL;
    }

    if (num_partitions <= 0) {
        printf("Why would you want less than 0 partitions?\n");
        return NULL;
    }

    #ifdef TIMING
        struct timespec t1 = get_time();
        struct timespec t2 = get_time();
        double elapsed;
    #endif

    omp_set_num_threads(num_threads);

    // Get the partition rule
    PartitionRule* partition_rule = malloc(sizeof(PartitionRule));
    create_partition(partition_rule, center_coords, num_nodes, num_partitions, frac_coords);
    // printf("Dim max: %f. Dim min: %f\n", partition_rule->max_val, partition_rule->min_val);
    PRINTF("Dim max: %f. Dim min: %f\n", partition_rule->max_val, partition_rule->min_val);
    PRINTF("First wall position: %f\n", partition_rule->walls[0]);

    // Check if the partition width is sufficient given the # of partitions
    int partition_size_flag = check_partition_size(num_partitions, partition_rule, lattice, atom_cutoff, bond_cutoff, use_bond_graph);

    // Allocate memory to store pointers to Partition structs (# of pointers = num_partitions)
    Partition** partitions = malloc(sizeof(Partition*) * num_partitions);
    for (unsigned int i = 0; i < num_partitions; i++) {
        partitions[i] = malloc(sizeof(Partition));

        initialize_empty_partition(partitions[i], i, num_edges, num_nodes, num_partitions);
    }

    PRINTF("Created partition rule\n");
    PRINTF("Dim to use (based on cartesian coordinates): %u\n", partition_rule->dim_to_use);
    PRINTF("Atom cutoff: %f\n", atom_cutoff);
    PRINTF("Number of bonds within atom radius: %ld\n", num_edges);
    PRINTF("Number of bonds within bond radius: %ld\n", num_within_bond_r_indices);
    PRINTF("USE_BOND_GRAPH IS %d\n", use_bond_graph);

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("TIMING: Created partitions and partition rule: %.9f\n", elapsed);

        t1 = get_time();
    #endif

    // Assign nodes to their respective to/from/pure bucket and create to_partition array
    nodes_to_partition = malloc(sizeof(int) * num_nodes); // see comment above global var declaration

    // TODO: this is the official new assign_to_partitions function. remove all references of "test"
    assign_to_partitions_test_2(partition_rule, partitions, num_partitions, num_nodes, frac_coords, num_edges, src_nodes, dst_nodes, num_threads);

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("TIMING: Assigned each node to a partition: %f\n", elapsed);

        t1 = get_time();
    #endif

    // Calculate the # of nodes per partition
    long* num_atoms_per_partition = malloc(sizeof(long) * num_partitions);
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        num_atoms_per_partition[partition_i] = get_total_num_atoms(partitions[partition_i], num_partitions);

        // TODO: testing, remove when done
        if (num_atoms_per_partition[partition_i] == 0) {
            printf("Num atoms per partition is zero?? Partition walls: %f. Max val, min val: %f %f\n", partition_rule->walls[0], partition_rule->min_val, partition_rule->max_val);
        }
    }

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("Calculated # of nodes per partition: %f\n", elapsed);

        t1 = get_time();
    #endif

    // Create global ID array for each partition
    long** global_id_arrays = malloc(sizeof(long*) * num_partitions); // Array of global ID arrays
    long** global_id_markers = malloc(sizeof(long*) * num_partitions); // Markers of indices where each segment begins (inclusive)

    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        global_id_arrays[partition_i] = malloc(sizeof(long) * num_atoms_per_partition[partition_i]);
        global_id_markers[partition_i] = malloc(sizeof(long) * (2 * num_partitions + 1));

        create_global_id_array(partitions[partition_i], global_id_arrays[partition_i], global_id_markers[partition_i], num_partitions);
    }

    // For each edge, assign the edge to the partition that the dst node is in
    // For bond graph, also create G2L_DE_mappings
    unsigned int dst_node_partition;
    long** G2L_DE_mappings = use_bond_graph ? malloc(sizeof(long*) * num_partitions) : NULL;

    if (use_bond_graph) {
        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
            G2L_DE_mappings[partition_i] = malloc(sizeof(long) * num_edges);
        }
    }

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        if (use_bond_graph) {
            printf("TIMING: Created global ID array, created global to local edge mappings: %f\n", elapsed);
        } else {
            printf("TIMING: Created global ID array: %f\n", elapsed);
        }

        t1 = get_time();
    #endif

    // Set up preliminaries to parallelize edge assignment -----------------------------------
    int start_ids[num_threads];
    int num_iters_array[num_threads];
    int quotient = num_edges / num_threads;
    int remainder = num_edges % num_threads;

    // Populate the start_ids and num_iters_array
    for (int i = 0; i < num_threads; i++) {
        start_ids[i] = i * quotient + (i < remainder ? i : remainder);
        num_iters_array[i] = quotient + (i < remainder ? 1 : 0);
    }

    // Iterate once to calculate prefix array
    int local_counts_array[num_partitions][num_threads];
    int prefix_array[num_partitions][num_threads];
    long max_counts[num_partitions];
    // Note: local_counts_array and prefix_array are arrays of points to arrays. This is because
    // we need to calculate local_count for each partition. This next loop populates the arrays

    // Restructured code to process each edge once

    // First pass: Calculate local counts per partition per thread
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int *local_counts = calloc(num_partitions, sizeof(int)); // Initialize to 0

        for (long edge_i = start_ids[thread_id]; edge_i < start_ids[thread_id] + num_iters_array[thread_id]; edge_i++) {
            int dst_node_partition = which_partition(partition_rule, &frac_coords[3 * dst_nodes[edge_i]]);
            local_counts[dst_node_partition]++;
        }

        // Store local counts into the shared array
        for (int p = 0; p < num_partitions; p++) {
            local_counts_array[p][thread_id] = local_counts[p];
        }
        free(local_counts);
    }

    // Compute prefix sums (same as before)
    int running_sum;
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        running_sum = 0;
        for (int thread_i = 0; thread_i < num_threads; thread_i++) {
            prefix_array[partition_i][thread_i] = running_sum;
            running_sum += local_counts_array[partition_i][thread_i];
        }
        max_counts[partition_i] = running_sum;
    }

    // Second pass: Assign edges to partitions
    #pragma omp parallel num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();
        int *local_offsets = calloc(num_partitions, sizeof(int)); // Track current offset per partition

        for (long edge_i = start_ids[thread_id]; edge_i < start_ids[thread_id] + num_iters_array[thread_id]; edge_i++) {
            int dst_node_partition = which_partition(partition_rule, &frac_coords[3 * dst_nodes[edge_i]]);
            int position = prefix_array[dst_node_partition][thread_id] + local_offsets[dst_node_partition];
            partitions[dst_node_partition]->edges_ids[position] = edge_i;

            if (use_bond_graph) {
                G2L_DE_mappings[dst_node_partition][edge_i] = position;
            }
            local_offsets[dst_node_partition]++;
        }
        free(local_offsets);
    }

    // Set partition edge counts
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        partitions[partition_i]->num_edges = max_counts[partition_i];
    }

    // First pass to calculate local counts
    // for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
    //     #pragma omp parallel private(dst_node_partition) num_threads(num_threads)
    //     {
    //         int thread_id = omp_get_thread_num();
    //         int local_count = 0;

    //         for (long edge_i = start_ids[thread_id]; edge_i < start_ids[thread_id] + num_iters_array[thread_id]; edge_i++) {
    //             dst_node_partition = which_partition(partition_rule, &frac_coords[3 * dst_nodes[edge_i]]);

    //             if (dst_node_partition == partition_i) {
    //                 local_count += 1;
    //             }
    //         }

    //         local_counts_array[partition_i][thread_id] = local_count;
    //     }
    // }

    // // Add up local counts to calculate prefix array
    // int running_sum;
    // for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
    //     running_sum = 0;
    //     for (int thread_i = 0; thread_i < num_threads; thread_i++) {
    //         prefix_array[partition_i][thread_i] = running_sum;
    //         running_sum += local_counts_array[partition_i][thread_i];
    //     }
    //     max_counts[partition_i] = running_sum;
    // }

    // for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
    //     #pragma omp parallel private(dst_node_partition) num_threads(num_threads)
    //     {
    //         int thread_id = omp_get_thread_num();
    //         int local_count = 0;

    //         for (long edge_i = start_ids[thread_id]; edge_i < start_ids[thread_id] + num_iters_array[thread_id]; edge_i++) {
    //             dst_node_partition = which_partition(partition_rule, &frac_coords[3 * dst_nodes[edge_i]]);

    //             if (dst_node_partition == partition_i) {
    //                 // Add to partitions edges_ids array
    //                 // partitions[partition_i]->edges_ids[partitions[partition_i]->num_edges] = edge_i;
    //                 partitions[partition_i]->edges_ids[prefix_array[partition_i][thread_id] + local_count] = edge_i;

    //                 if (use_bond_graph) {
    //                     G2L_DE_mappings[partition_i][edge_i] = prefix_array[partition_i][thread_id] + local_count;
    //                 }
    //                 local_count += 1;
    //             }
    //         }
    //     }
    //     partitions[partition_i]->num_edges = max_counts[partition_i];
    // }

    // The above section is the edge assignment code (parallelized) -----------------------------------
    // Here is the original, unparallelized loop. Kept here for reference. Parallelism is hard!
    // double* dst_cords;
    // for (long edge_i = 0; edge_i < num_edges; edge_i++) {
    //     dst_coords = &frac_coords[3 * dst_nodes[edge_i]];
    //     dst_node_partition = which_partition(partition_rule, dst_coords);


    //     partitions[dst_node_partition]->edges_ids[partitions[dst_node_partition]->num_edges] = edge_i;

    //     if (use_bond_graph) {
    //         G2L_DE_mappings[dst_node_partition][edge_i] = partitions[dst_node_partition]->num_edges;
    //     }

    //     partitions[dst_node_partition]->num_edges += 1;
    // }

    if (use_bond_graph) {
        // Allocating memory for each partition's pure_UDEs field
        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
            partitions[partition_i]->pure_UDEs = malloc(sizeof(BDirectedEdge*) * partitions[partition_i]->num_edges);
        }
    }

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("TIMING: edge assignment time: %f\n", elapsed);

        t1 = get_time();
    #endif


    // For each partition, create a mapping from global to local indices
    // mapping is represented by a contiguous block of memory where the ith index represents
    // the local index for the ith global index node
    long** G2L_mappings = malloc(sizeof(long*) * num_partitions); // num_partitions pointers to various arrays
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        G2L_mappings[partition_i] = malloc(sizeof(long) * num_nodes);

        // set all values within the G2L_mappings to be -1 for now
        for (long node_i = 0; node_i < num_nodes; node_i++) {
            G2L_mappings[partition_i][node_i] = -1;
        }

        // Create mapping from global to local
        for (long node_i = 0; node_i < num_atoms_per_partition[partition_i]; node_i++) {
            G2L_mappings[partition_i][global_id_arrays[partition_i][node_i]] = node_i;
        }
    }

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("TIMING: Mapping creation time: %f\n", elapsed);

        t1 = get_time();
    #endif

    // Using the prior mapping, convert global edges array to local edges array for each partition
    long** local_edges_src_nodes = malloc(sizeof(long*) * num_partitions);
    long** local_edges_dst_nodes = malloc(sizeof(long*) * num_partitions);

    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        local_edges_src_nodes[partition_i] = malloc(sizeof(long) * partitions[partition_i]->num_edges);
        local_edges_dst_nodes[partition_i] = malloc(sizeof(long) * partitions[partition_i]->num_edges);
    }

    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        #pragma omp parallel for schedule(static, partitions[partition_i]->num_edges / num_threads) num_threads(num_threads)
        for (long edge_i = 0; edge_i < partitions[partition_i]->num_edges; edge_i++) {
            // Find the local node index for the global src edge node
            local_edges_src_nodes[partition_i][edge_i] = G2L_mappings[partition_i][src_nodes[partitions[partition_i]->edges_ids[edge_i]]];
            local_edges_dst_nodes[partition_i][edge_i] = G2L_mappings[partition_i][dst_nodes[partitions[partition_i]->edges_ids[edge_i]]];

            #ifdef DEBUG
                // Find the global node ids of the nodes that show up as -1
                int num_missing = 0;
                if (local_edges_src_nodes[partition_i][edge_i] == -1) {
                    printf("SRC node in partition %u with global index %ld found to be -1\n", partition_i, src_nodes[partitions[partition_i]->edges_ids[edge_i]]);
                    printf("This node's position is: %f, %f, %f\n", frac_coords[3 * src_nodes[partitions[partition_i]->edges_ids[edge_i]]],
                                                            frac_coords[1 + (3 * src_nodes[partitions[partition_i]->edges_ids[edge_i]])],
                                                            frac_coords[2 + (3 * src_nodes[partitions[partition_i]->edges_ids[edge_i]])]);

                    printf("corresponding DST node location: %f %f %f\n", frac_coords[3 * dst_nodes[partitions[partition_i]->edges_ids[edge_i]]],
                                                            frac_coords[1 + (3 * dst_nodes[partitions[partition_i]->edges_ids[edge_i]])],
                                                            frac_coords[2 + (3 * dst_nodes[partitions[partition_i]->edges_ids[edge_i]])]);
                    printf("FPIS distance for this edge: %f\n", distances[partitions[partition_i]->edges_ids[edge_i]]);
                    num_missing += 1;
                }

                if (local_edges_dst_nodes[partition_i][edge_i] == -1) {
                    printf("DST node in partition %u with global index %ld found to be -1\n", partition_i, dst_nodes[partitions[partition_i]->edges_ids[edge_i]]);
                    printf("This node's position is: %f, %f, %f\n", frac_coords[3 * dst_nodes[partitions[partition_i]->edges_ids[edge_i]]],
                                                            frac_coords[1 + 3 * dst_nodes[partitions[partition_i]->edges_ids[edge_i]]],
                                                            frac_coords[2 + 3 * dst_nodes[partitions[partition_i]->edges_ids[edge_i]]]);
                    num_missing += 1;
                }

                if (num_missing != 0) {
                    printf("Num missing for this edge in partition %u is %d\n", partition_i, num_missing);
                }
            #endif
        }
    }

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("TIMING: Mapping usage time: %f\n", elapsed);

        t1 = get_time();
    #endif

    // Create bond graph ---------------------------------------------------------
    long** bond_mapping_DE;
    long** bond_mapping_BDE;
    long* num_bond_mapping;

    long max_num_BDEs_in_a_partition;

    AdjList* adj_lists;
    AdjListEntry* this_entry;
    BDirectedEdge* this_BDE;

    long** BDE_marker_arrays;
    long** G2L_BDE_mappings;

    long** local_bond_mapping_DE;
    long** local_bond_mapping_BDE;

    long** line_src_nodes;
    long** line_dst_nodes;
    long** center_atom_indices;
    long* line_num_edges;

    // Initialize 1 AdjList for each gpu
    if (use_bond_graph) {

        adj_lists = malloc(sizeof(AdjList) * num_partitions);

        bond_mapping_DE = malloc(sizeof(long*) * num_partitions); // array of arrays of GLOBAL DE ids
        bond_mapping_BDE = malloc(sizeof(long*) * num_partitions); // array of arrays of "GLOBAL" BDE ids
        num_bond_mapping = malloc(sizeof(long) * num_partitions); // array of size num_partitions depicting the # of elements in each partition's bond_mapping

        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
            adj_lists[partition_i].array_of_entries = malloc(sizeof(AdjListEntry) * num_nodes);
            adj_lists[partition_i].num_entries = num_nodes;

            for (long entry_i = 0; entry_i < num_nodes; entry_i++) {
                // TODO: do one malloc instead of num_nodes amount of mallocs - HIGH PRIORITY
                adj_lists[partition_i].array_of_entries[entry_i].edges = malloc(sizeof(BDirectedEdge*) * DEFAULT_ITEMS);
                adj_lists[partition_i].array_of_entries[entry_i].num_edges = 0;
            }

            bond_mapping_DE[partition_i] = malloc(sizeof(long) * partitions[partition_i]->num_edges);
            bond_mapping_BDE[partition_i] = malloc(sizeof(long) * partitions[partition_i]->num_edges);

            num_bond_mapping[partition_i] = 0;
        }

        #ifdef TIMING
            t2 = get_time();
            elapsed = time_diff(t1, t2);
            PRINTF("TIMING: Malloc for adjlist time: %f\n", elapsed);

            t1 = get_time();
        #endif

        // Create psuedo-adjacency list for each partition.
        // First iterate through all edges w/ distance <= bond_r + TOL and determine if the edge belongs to our current GPU
        // TODO: this can be parallelized across multiple GPUs at the very least (if not massively parallelized with mutexes)
        long src_node;
        long dst_node;

        PartitionTransferInfo* partition_transfer_info_tmp;

        long curr_BDE_id;
        max_num_BDEs_in_a_partition = 0;

        // First, iterate through all DEs of distance <= bond_r -- ONLY considering DEs that will become border BDEs for now

        // Border edges are determined by whether or not the dst node is within a border region
        // Note that it's possible for dst to be in multiple atom border regions. In this case, nodes_to_partition will only store 1
        // border region that dst is in. However, I believe that, in the case where this happens, src will never be in a pure region AND less than bond_r
        // distance away anyways (so we don't have to worry about that case). This assumes bond_r <= atom_r. If there are any weird behaviours, it's probably
        // coming from this assumption lol.
        // TODO: if you want to extend this code to work with non-vertical-line partitions, then you have to change something about this assumption

        // Further, there exist border edges whose dst AND src nodes are within a particular border region. These BDEs are not computed by this partition;
        // however, they are necessary to fully compute the features of the edge that we do care about. These BDEs point from border node to border node
        for (int partition_i = 0; partition_i < (int) num_partitions; partition_i++) {
            curr_BDE_id = 0;

            for (long i = 0; i < num_within_bond_r_indices; i++) {

                src_node = src_nodes[within_bond_r_indices[i]];
                dst_node = dst_nodes[within_bond_r_indices[i]];

                // We do not handle aliased edges as they should not exist within large systems
                if (src_node == dst_node) {
                    printf("ERROR: Found aliased edge (self-loop where node has edge pointing to itself). This shouldn't happen if the unit cell is larger than the bond graph cutoff. Results may not be correct.");
                }

                if (G2L_mappings[partition_i][dst_node] != -1) {
                    // This conditional is true if this dst_node exists somewhere within the current partition (including border nodes/atoms)

                    if (nodes_to_partition[dst_node] == partition_i) {
                        // This conditional is true if the DE is a DE that we need FROM another partition

                        // Create BDE
                        this_BDE = malloc(sizeof(BDirectedEdge)); // TODO: don't need to malloc one at a time...
                        this_BDE->offset = &images[3 * within_bond_r_indices[i]];
                        this_BDE->init_src_node = src_node;
                        this_BDE->init_dst_node = dst_node;
                        this_BDE->global_id = curr_BDE_id;
                        this_BDE->needs_in_line = false;

                        curr_BDE_id += 1;

                        // Add this BDE to the "from" PartitionTransferInfo
                        partition_transfer_info_tmp = partitions[partition_i]->transfer_info[which_partition(partition_rule, &frac_coords[3 * dst_node])];
                        partition_transfer_info_tmp->edges_from_gpu[partition_transfer_info_tmp->num_edges_from_gpu] = this_BDE;
                        partition_transfer_info_tmp->num_edges_from_gpu += 1;

                        if (partition_transfer_info_tmp->num_edges_from_gpu % DEFAULT_ITEMS == 0) {
                            //printf("HIT transfer info border edge realloc. num edges from gpu: %ld\n", partition_transfer_info_tmp->num_edges_from_gpu); // TODO: it looks like this is happening a lot!!!! try usinga different DEFAULT_ITEMS that scales with the # of atoms
                            partitions[partition_i]->transfer_info[which_partition(partition_rule, &frac_coords[3 * dst_node])]->edges_from_gpu = realloc(partitions[partition_i]->transfer_info[which_partition(partition_rule, &frac_coords[3 * dst_node])]->edges_from_gpu, sizeof(BDirectedEdge*) * (partition_transfer_info_tmp->num_edges_from_gpu + DEFAULT_ITEMS));
                            if (partitions[partition_i]->transfer_info[which_partition(partition_rule, &frac_coords[3 * dst_node])]->edges_from_gpu == NULL) {
                                printf("ERROR: realloced to NULL. You need more memory bruh\n");
                            }
                        }

                        // Add this new BDE to the edge table (but not to any of the bond_mapping arrays)
                        this_entry = &adj_lists[partition_i].array_of_entries[src_node];

                        this_entry->edges[this_entry->num_edges] = this_BDE;
                        this_entry->num_edges += 1;

                        if (this_entry->num_edges % DEFAULT_ITEMS == 0) {
                            PRINTF("reallocing edges within array of entries\n");
                            adj_lists[partition_i].array_of_entries[src_node].edges = realloc(adj_lists[partition_i].array_of_entries[src_node].edges, sizeof(BDirectedEdge*) * (this_entry->num_edges + DEFAULT_ITEMS));
                        }

                    } else if (nodes_to_partition[dst_node] != partition_i && nodes_to_partition[dst_node] != -1) {
                        // This conditional is true if the DE is a DE that we're sending TO another partition
                        // NOTE: this means the BDE in the "to" section points to a different BDE struct than the same BDE in the corresponding "from" section. This makes sense
                        // since each UDE struct should have different local ids depending on which partition they're a part of

                        // Create the UDE struct
                        this_BDE = malloc(sizeof(BDirectedEdge)); // TODO: don't need to malloc one at a time...
                        this_BDE->offset = &images[3 * within_bond_r_indices[i]];
                        this_BDE->init_src_node = src_node;
                        this_BDE->init_dst_node = dst_node;
                        this_BDE->global_id = curr_BDE_id;
                        this_BDE->needs_in_line = true;

                        curr_BDE_id += 1;

                        // Add to pertinent PartitionTransferInfo struct
                        partition_transfer_info_tmp = partitions[partition_i]->transfer_info[nodes_to_partition[dst_node]];
                        partition_transfer_info_tmp->edges_to_gpu[partition_transfer_info_tmp->num_edges_to_gpu] = this_BDE;
                        partition_transfer_info_tmp->num_edges_to_gpu += 1;

                        if (partition_transfer_info_tmp->num_edges_to_gpu % DEFAULT_ITEMS == 0) {
                            // TODO: it looks like this is happening a lot!!!! try using a different DEFAULT_ITEMS that scales with the # of atoms
                            partitions[partition_i]->transfer_info[nodes_to_partition[dst_node]]->edges_to_gpu = realloc(partitions[partition_i]->transfer_info[nodes_to_partition[dst_node]]->edges_to_gpu, sizeof(BDirectedEdge*) * (partition_transfer_info_tmp->num_edges_to_gpu + DEFAULT_ITEMS));
                            if (partitions[partition_i]->transfer_info[nodes_to_partition[dst_node]]->edges_to_gpu == NULL) {
                                printf("ERROR: realloced to NULL\n");
                            }
                        }

                        // Add to this partition's edge table
                        this_entry = &adj_lists[partition_i].array_of_entries[src_node];

                        this_entry->edges[this_entry->num_edges] = this_BDE;
                        this_entry->num_edges += 1;

                        if (this_entry->num_edges % DEFAULT_ITEMS == 0) {
                            PRINTF("reallocing edges within array of entries\n");
                            adj_lists[partition_i].array_of_entries[src_node].edges = realloc(adj_lists[partition_i].array_of_entries[src_node].edges, sizeof(BDirectedEdge*) * (this_entry->num_edges + DEFAULT_ITEMS));
                        }

                        // Add this DE/BDE pair to the bond_mapping_DE and bond_mapping_BDE arrays
                        bond_mapping_DE[partition_i][num_bond_mapping[partition_i]] = within_bond_r_indices[i];
                        bond_mapping_BDE[partition_i][num_bond_mapping[partition_i]] = this_BDE->global_id;

                        num_bond_mapping[partition_i] += 1;
                    } else {
                        continue;
                    }
                }
            }

            // Now, iterate a second time through all of the global DEs with distance <= bond_r. This time, only focus on pure UDEs
            for (long i = 0; i < num_within_bond_r_indices; i++) {
                src_node = src_nodes[within_bond_r_indices[i]];
                dst_node = dst_nodes[within_bond_r_indices[i]];

                if (G2L_mappings[partition_i][dst_node] != -1) {

                    if (nodes_to_partition[dst_node] == partition_i) {
                        continue;
                    } else if (nodes_to_partition[dst_node] != partition_i && nodes_to_partition[dst_node] != -1) {
                        continue;
                    } else if (which_partition(partition_rule, &frac_coords[3 * dst_node]) == (unsigned int) partition_i) {
                        // This DE is not a border DE from another partition, nor is it a border DE to another partition

                        this_entry = &adj_lists[partition_i].array_of_entries[src_node];

                        // Create a new BDE and assign it to the correct entry. Also, add the BDE to the pertinent Partition struct. Also, add to bond_mapping arrays
                        this_BDE = malloc(sizeof(BDirectedEdge)); // TODO: don't need to do this every single time, should just always malloc once
                        this_BDE->offset = &images[3 * within_bond_r_indices[i]];
                        this_BDE->init_src_node = src_node;
                        this_BDE->init_dst_node = dst_node;
                        this_BDE->global_id = curr_BDE_id;
                        this_BDE->needs_in_line = true;

                        curr_BDE_id += 1;

                        adj_lists[partition_i].array_of_entries[src_node].edges[this_entry->num_edges] = this_BDE;
                        adj_lists[partition_i].array_of_entries[src_node].num_edges += 1;

                        if (adj_lists[partition_i].array_of_entries[src_node].num_edges % DEFAULT_ITEMS == 0) {
                            PRINTF("reallocing entry %ld\n", src_node);
                            adj_lists[partition_i].array_of_entries[src_node].edges = realloc(adj_lists[partition_i].array_of_entries[src_node].edges, sizeof(BDirectedEdge*) * (adj_lists[partition_i].array_of_entries[src_node].num_edges + DEFAULT_ITEMS));
                        }

                        // Add this BDE to the pertinent Partition struct
                        partitions[partition_i]->pure_UDEs[partitions[partition_i]->num_UDEs] = this_BDE;
                        partitions[partition_i]->num_UDEs += 1;

                        // Update bond_mapping_DE and bond_mapping_UDE
                        bond_mapping_DE[partition_i][num_bond_mapping[partition_i]] = within_bond_r_indices[i];
                        bond_mapping_BDE[partition_i][num_bond_mapping[partition_i]] = this_BDE->global_id;

                        num_bond_mapping[partition_i] += 1;

                    }
                    else {
                        continue;
                    }
                }
            }

            max_num_BDEs_in_a_partition = max_long(curr_BDE_id, max_num_BDEs_in_a_partition);
            PRINTF("BDE global id after complete iteration through all BDEs: %ld \n", curr_BDE_id);
        }

        #ifdef TIMING
            t2 = get_time();
            elapsed = time_diff(t1, t2);
            PRINTF("TIMING: Created edge table time: %f\n", elapsed);

            t1 = get_time();
        #endif

        // For each partition, create local ids for BDEs (and assign to BDE structs), create markers array for bond graph, localize bond_mapping_DE using the G2L_edge_mappings
        // WARNING: BDirectedEdge->local_id is uninitialized at this point in the code
        // TODO: this can be parallelized across GPUs
        BDE_marker_arrays = malloc(sizeof(long*) * num_partitions);
        G2L_BDE_mappings = malloc(sizeof(long*) * num_partitions);

        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
            // Allocate BDE_marker_array for this partition
            BDE_marker_arrays[partition_i] = malloc(sizeof(long) * (2 * num_partitions + 1));
            G2L_BDE_mappings[partition_i] = malloc(sizeof(long) * max_num_BDEs_in_a_partition);

            localize_UDE_and_create_UDE_markers(partitions[partition_i], num_partitions, BDE_marker_arrays[partition_i], G2L_BDE_mappings[partition_i]);
        }
        PRINTF("Localized BDE and created BDE markers\n");

        // Convert bond_mappings into local_bond_mapping_DE and local_bond_mapping_UDE
        local_bond_mapping_DE = malloc(sizeof(long*) * num_partitions);
        local_bond_mapping_BDE = malloc(sizeof(long*) * num_partitions);

        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
            local_bond_mapping_DE[partition_i] = malloc(sizeof(long) * num_bond_mapping[partition_i]);
            local_bond_mapping_BDE[partition_i] = malloc(sizeof(long) * num_bond_mapping[partition_i]);

            for (long bond_i = 0; bond_i < num_bond_mapping[partition_i]; bond_i++) {
                local_bond_mapping_DE[partition_i][bond_i] = G2L_DE_mappings[partition_i][bond_mapping_DE[partition_i][bond_i]];
                local_bond_mapping_BDE[partition_i][bond_i] = G2L_BDE_mappings[partition_i][bond_mapping_BDE[partition_i][bond_i]];
            }
        }

        PRINTF("Converted global bond mappings into local bond mappings\n");

        line_src_nodes = malloc(sizeof(long*) * num_partitions);
        line_dst_nodes = malloc(sizeof(long*) * num_partitions);
        center_atom_indices = malloc(sizeof(long*) * num_partitions); //  the global indices of the atoms (nodes in atom graph) that are the center nodes for this line. gets converted into local center atom indices later
        line_num_edges = malloc(sizeof(long) * num_partitions);

        AdjListEntry* second_entry;

        PRINTF("Creating line graph\n");
        // Create line graph from edge list
        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {

            line_src_nodes[partition_i] = malloc(sizeof(long) * LINE_GRAPH_DEFAULT_ITEMS);
            line_dst_nodes[partition_i] = malloc(sizeof(long) * LINE_GRAPH_DEFAULT_ITEMS);
            center_atom_indices[partition_i] = malloc(sizeof(long) * LINE_GRAPH_DEFAULT_ITEMS);
            line_num_edges[partition_i] = 0;

            // Iterate through all entries in the adjacency table
            for (long entry_i = 0; entry_i < adj_lists[partition_i].num_entries; entry_i++) {
                this_entry = &adj_lists[partition_i].array_of_entries[entry_i];

                // Iterate through each UDE within the entry
                for (long UDE_i = 0; UDE_i < this_entry->num_edges; UDE_i++) {
                    this_BDE = this_entry->edges[UDE_i];

                    // Get the second entry (which should be whatever entry that isn't entry_i)
                    second_entry = &adj_lists[partition_i].array_of_entries[this_BDE->init_dst_node];

                    // For each UDE in the second_entry, draw a line to UDE_j
                    for (long UDE_j = 0; UDE_j < second_entry->num_edges; UDE_j++) {

                        // Draw a line from this_UDE to our second_entry's UDE if the second_entry's UDE needs an incoming line
                        if (second_entry->edges[UDE_j]->needs_in_line == true) {

                            if (second_entry->edges[UDE_j]->init_dst_node == this_BDE->init_src_node) {
                                continue;
                            }

                            line_src_nodes[partition_i][line_num_edges[partition_i]] = this_BDE->local_id;
                            line_dst_nodes[partition_i][line_num_edges[partition_i]] = second_entry->edges[UDE_j]->local_id;
                            center_atom_indices[partition_i][line_num_edges[partition_i]] = second_entry->edges[UDE_j]->init_src_node;

                            line_num_edges[partition_i] += 1;
                        }

                        if (line_num_edges[partition_i] != 0 && line_num_edges[partition_i] % LINE_GRAPH_DEFAULT_ITEMS == 0) {
                            PRINTF("Reallocing # of line graph items for partition %u\n", partition_i);
                            line_src_nodes[partition_i] = realloc(line_src_nodes[partition_i], sizeof(long) * (line_num_edges[partition_i] + LINE_GRAPH_DEFAULT_ITEMS));
                            line_dst_nodes[partition_i] = realloc(line_dst_nodes[partition_i], sizeof(long) * (line_num_edges[partition_i] + LINE_GRAPH_DEFAULT_ITEMS));
                            center_atom_indices[partition_i] = realloc(center_atom_indices[partition_i], sizeof(long) * (line_num_edges[partition_i] + LINE_GRAPH_DEFAULT_ITEMS));

                            if (line_src_nodes[partition_i] == NULL || line_dst_nodes[partition_i] == NULL || center_atom_indices[partition_i] == NULL) {
                                printf("FAILED REALLOC DURING LINE GRAPH Memory reallocation! You need more memory my dude.\n");
                            }
                        }
                    }
                }
            }
        }

        // For center_atom_indices, convert global node id to local node id (remember that these indices are nodes in the atom graph that are associated with the particular drawn line)
        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
            for (long line_i = 0; line_i < line_num_edges[partition_i]; line_i++) {
                if (G2L_mappings[partition_i][center_atom_indices[partition_i][line_i]] == -1) {
                    printf("ERROR ERROR ERROR: found global id in center_atom_indices that isn't in the global to local mapping for this partition");
                }
                center_atom_indices[partition_i][line_i] = G2L_mappings[partition_i][center_atom_indices[partition_i][line_i]];
            }
        }

    }


    #ifdef DEBUG
    if (use_bond_graph) {
        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
            PRINTF("num lines in partition %u: %ld\n", partition_i, line_num_edges[partition_i]);
        }


        PRINTF("first values in line src nodes: %ld, %ld, %ld, %ld\n", line_src_nodes[0][0], line_src_nodes[0][1], line_src_nodes[0][2], line_src_nodes[0][3]);
    }
    #endif

    #ifdef TIMING
        if (use_bond_graph) {
            t2 = get_time();
            elapsed = time_diff(t1, t2);
            printf("TIMING: Created line graph for each gpu: %f\n", elapsed);
        }

        t1 = get_time();
    #endif


    // ------------------------------------------------------------------------

    // Rearrange node position information using the global_array_id indices
    double** local_center_coords = malloc(sizeof(double*) * num_partitions); // The coordinates for nodes in each partition, rearranged to match global_id_arrays
    long global_index;

    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        local_center_coords[partition_i] = malloc(sizeof(double) * (3 * num_atoms_per_partition[partition_i]));

        for (long node_i = 0; node_i < num_atoms_per_partition[partition_i]; node_i++) {
            global_index = global_id_arrays[partition_i][node_i];

            local_center_coords[partition_i][3 * node_i] = center_coords[3 * global_index];
            local_center_coords[partition_i][3 * node_i + 1] = center_coords[3 * global_index + 1];
            local_center_coords[partition_i][3 * node_i + 2] = center_coords[3 * global_index + 2];
        }
    }

    long* num_UDEs_per_partition;

    if (use_bond_graph) {
        num_UDEs_per_partition = malloc(sizeof(long) * num_partitions);

        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
            num_UDEs_per_partition[partition_i] = get_num_UDEs_in_partition(partitions[partition_i], num_partitions);
        }
    }

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("TIMING: Misc end tasks: %f\n", elapsed);
    #endif

    Results* results = malloc(sizeof(Results));
    results->global_id_arrays = global_id_arrays;
    results->global_id_markers = global_id_markers;
    results->partitions = partitions;
    results->local_edge_src_nodes = local_edges_src_nodes;
    results->local_edge_dst_nodes = local_edges_dst_nodes;
    results->num_atoms_per_partition = num_atoms_per_partition;
    results->local_center_coords = local_center_coords;

    // line/bond graph items
    if (use_bond_graph) {
        results->line_src_nodes = line_src_nodes;
        results->line_dst_nodes = line_dst_nodes;
        results->center_atom_indices = center_atom_indices;
        results->line_num_edges = line_num_edges; // # of lines per partition
        results->num_UDEs_per_partition = num_UDEs_per_partition; // # of UDEs per partition

        results->UDE_marker_arrays = BDE_marker_arrays;

        results->local_bond_mapping_DE = local_bond_mapping_DE;
        results->local_bond_mapping_UDE = local_bond_mapping_BDE;
        results->num_bond_mapping = num_bond_mapping;

        results->G2L_DE_mappings = G2L_DE_mappings;// TODO: for debugging purposes. remove when complete. NOT RELIABLE
        // TODO: when removing G2L_DE_mappings, don't forget to free everything! (should be two free calls)
    }


    // Run through the local_edges_src_nodes and local_edges_dst_nodes to find -1 values TODO: testing, uncomment when done
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        for (long edge_i = 0; edge_i < partitions[partition_i]->num_edges; edge_i++) {
            if (local_edges_src_nodes[partition_i][edge_i] == -1) {
                printf("Found -1 src node in %u with edge index %ld\n", partition_i, edge_i);
            }
        }

        for (long edge_i = 0; edge_i < partitions[partition_i]->num_edges; edge_i++) {
            if (local_edges_dst_nodes[partition_i][edge_i] == -1) {
                printf("Found -1 dst node in %u with edge index %ld\n", partition_i, edge_i);
            }
        }
    }

    #ifdef DEBUG
    // Check if FPIS gave us src_nodes or dst_nodes with the value of -1
    for (unsigned int edge_i = 0; edge_i < num_edges; edge_i++) {
        if (src_nodes[edge_i] == -1) {
            printf("Found -1 src node within FPIS-given src_nodes\n");
        }

        if (dst_nodes[edge_i] == -1) {
            printf("Found -1 dst node within FPIS-given dst_nodes\n");
        }
    }
    #endif

    // Free memory
    free(partition_rule->walls);
    free(partition_rule);

    free(nodes_to_partition);

    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        free(G2L_mappings[partition_i]);
    }

    if (use_bond_graph) {
        for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
            for (long entry_i = 0; entry_i < num_nodes; entry_i++) {
                for (long edge_i = 0; edge_i < adj_lists[partition_i].array_of_entries[entry_i].num_edges; edge_i++) {
                    free(adj_lists[partition_i].array_of_entries[entry_i].edges[edge_i]);
                }

                free(adj_lists[partition_i].array_of_entries[entry_i].edges);

            }

            free(adj_lists[partition_i].array_of_entries);

            free(bond_mapping_DE[partition_i]);
            free(bond_mapping_BDE[partition_i]);

            free(G2L_BDE_mappings[partition_i]);
        }
    }

    free(G2L_mappings);

    if (use_bond_graph) {
        free(bond_mapping_BDE);
        free(bond_mapping_DE);

        free(G2L_BDE_mappings);

        free(adj_lists);
    }

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("TIMING: freed some stuff: %f\n", elapsed);

        struct timespec tf = get_time();
        elapsed = time_diff(t0, tf);
        printf("TIMING: c-level subgraph creation time: %f\n", elapsed);
    #endif

    return results;
}


/*
Adds to_add UDE* to entry
*/
// void add_UDE_to_entry(AdjListEntry* entry, UndirectedEdge* to_add) {
//     entry->edges[entry->num_edges] = to_add;
//     entry->num_edges += 1;

//     if (entry->num_edges % DEFAULT_ITEMS == 0) {
//         entry->edges = realloc(entry->edges, sizeof(UndirectedEdge*) * (entry->num_edges + DEFAULT_ITEMS));

//         if (entry->edges == NULL) {
//             printf("MEMORYERROR: not enough memory to add UDE to the current entry in our table\n");
//         }

//     }
// }

/*
Gets the total number of UDEs (nodes in bond graph) in a given partition.
*/
long get_num_UDEs_in_partition(Partition* partition, unsigned int num_partitions) {
    long num_UDEs = partition->num_UDEs;

    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        if (partition_i == partition->gpu_id) {
            continue;
        } else {
            num_UDEs += partition->transfer_info[partition_i]->num_edges_to_gpu;
            num_UDEs += partition->transfer_info[partition_i]->num_edges_from_gpu;
        }
    }

    return num_UDEs;
}

/*
Given a partition and BDE_marker_array pointer (that's already been malloc'd), assign the local_id field for all BDEs within a partition,
including border BDEs within PartitionTransferInfo structs. Also, populate the BDE_marker_array with the relevant items
*/
void localize_UDE_and_create_UDE_markers(Partition* partition, unsigned int num_partitions, long* BDE_marker_array, long* G2L_BDE_mapping) {
    long index = 0;
    long marker_index = 0;

    // Begin with pure BDEs
    BDE_marker_array[marker_index] = index;
    marker_index += 1;

    // First, work through the pure BDEs
    BDirectedEdge* tmp;
    for (long UDE_i = 0; UDE_i < partition->num_UDEs; UDE_i++) {
        tmp = partition->pure_UDEs[UDE_i];
        tmp->local_id = index;

        G2L_BDE_mapping[tmp->global_id] = tmp->local_id;

        index += 1;
    }

    // Next, assign to_gpu UDEs
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        if (partition_i == partition->gpu_id) {
            BDE_marker_array[marker_index] = index;
            marker_index += 1;
        } else {
            BDE_marker_array[marker_index] = index;
            marker_index += 1;

            for (long BDE_i = 0; BDE_i < partition->transfer_info[partition_i]->num_edges_to_gpu; BDE_i++) {
                tmp = partition->transfer_info[partition_i]->edges_to_gpu[BDE_i];
                tmp->local_id = index;

                G2L_BDE_mapping[tmp->global_id] = tmp->local_id;

                index += 1;
            }
        }
    }

    // Assign local_ids for from_gpu UDEs now
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        if (partition_i == partition->gpu_id) {
            BDE_marker_array[marker_index] = index;
            marker_index += 1;
        } else {
            BDE_marker_array[marker_index] = index;
            marker_index += 1;

            for (long UDE_i = 0; UDE_i < partition->transfer_info[partition_i]->num_edges_from_gpu; UDE_i++) {
                tmp = partition->transfer_info[partition_i]->edges_from_gpu[UDE_i];
                tmp->local_id = index;

                G2L_BDE_mapping[tmp->global_id] = tmp->local_id;

                index += 1;
            }
        }
    }
}

/*
Returns true if this DE is related to UDE (DE should be the opposite of the DE used to initialize UDE)
*/
// bool DE_UDE_equivalent(UndirectedEdge* UDE, long DE_src_node, long DE_dst_node, double* DE_offset) {
//     if
//     (
//         DE_src_node == UDE->init_src_node &&
//         DE_dst_node == UDE->init_dst_node &&
//         close_enough(DE_offset[0], UDE->offset[0]) &&
//         close_enough(DE_offset[1], UDE->offset[1]) &&
//         close_enough(DE_offset[2], UDE->offset[2])
//     ) {
//         printf("ERROR!!! The directed edge is the same as the initializing directed edge for this UDE. THIS SHOULD NEVER HAPPEN!\n");
//         return true;
//     }

//     if
//     (
//         DE_src_node == UDE->init_dst_node &&
//         DE_dst_node == UDE->init_src_node &&
//         close_enough(DE_offset[0], -1 * UDE->offset[0]) &&
//         close_enough(DE_offset[1], -1 * UDE->offset[1]) &&
//         close_enough(DE_offset[2], -1 * UDE->offset[2])
//     ) {
//         if (UDE->num_corresponding_DEs != 1) {
//             printf("ERROR: correct corresponding UDE already has %ld linked DE edges. This shouldn't happen.\n", UDE->num_corresponding_DEs);
//         }
//         return true;
//     }

//     return false;
// }

/*
Checks whether or not two doubles are close enough together that we can simply call them the same value
*/
bool close_enough(double a, double b) {
    if (a <= b + NUMERICAL_TOL && a >= b - NUMERICAL_TOL) {
        return true;
    }

    return false;
}


unsigned long hash_function(long x, long y) {
    //return (unsigned long) ((x + y)*(x + y + 1)/2) + y;
    unsigned int int1 = (unsigned int) x;
    unsigned long int2 = y << 32;
    unsigned long key = int1 + int2;

    key = (~key) + (key << 21); // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8); // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4); // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
}




/*
Creates the global id arary for the partition. Global id array is of the form: [pure_dst, to_0, to_1, ... from_0, from _1...]
The global_id_marker's values are the indices in which the corresponding region begins. So if global_id_marker were [0, 3, 3, 5, 5],
then the pure_dst region goes from [0, 3), there is no to_0 region (this must be gpu 0 then), to_1 is [3, 5), from_0 is (5, 5), from_1 is [5, len(marker)]
*/
void create_global_id_array(Partition* partition, long* global_id_array, long* global_id_marker, unsigned int num_partitions){
    long index = 0;
    long marker_index = 0;

    // Begin with pure dst
    global_id_marker[marker_index] = index;
    marker_index += 1;

    // First assign, pure dst nodes
    for (long dst_i = 0; dst_i < partition->num_pure_dst_nodes; dst_i++) {
        global_id_array[index] = partition->pure_dst_nodes[dst_i];

        index += 1;
    }

    // Assign to_gpu nodes to the global ID array first
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {

        if (partition_i == partition->gpu_id) {
            global_id_marker[marker_index] = index;
            marker_index += 1;
        } else {
            // Start with to_gpu
            global_id_marker[marker_index] = index;
            marker_index += 1;

            for (long node_i = 0; node_i < partition->transfer_info[partition_i]->num_to_gpu; node_i++) {
                global_id_array[index] = partition->transfer_info[partition_i]->to_gpu[node_i];

                index += 1;
            }
        }
    }

    // Assign from_gpus nodes to the global ID array second
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        if (partition_i == partition->gpu_id) {
            global_id_marker[marker_index] = index;
            marker_index += 1;
        } else {
            // Move to from_gpu
            global_id_marker[marker_index] = index;
            marker_index += 1;

            for (long node_i = 0; node_i < partition->transfer_info[partition_i]->num_from_gpu; node_i++) {
                global_id_array[index] = partition->transfer_info[partition_i]->from_gpu[node_i];

                index += 1;
            }
        }
    }

}

/*
Returns the total number of atoms in a partition (including dst and src nodes)
*/
long get_total_num_atoms(Partition* partition, unsigned int num_partitions) {
    long total = partition->num_pure_dst_nodes;

    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        if (partition_i != partition->gpu_id) {
            total += partition->transfer_info[partition_i]->num_from_gpu;
            total += partition->transfer_info[partition_i]->num_to_gpu;
        }
    }

    return total;
}

void print_partitions(Partition** partitions, unsigned int num_partitions) {
    for (unsigned int partition_i = 0; partition_i < num_partitions; partition_i++) {
        printf("Partition %u has %lu pure nodes.\n", partition_i, partitions[1]->num_pure_dst_nodes);

        for (unsigned int partition_j = 0; partition_j < num_partitions; partition_j++) {
            if (partition_i != partition_j) {
                printf("\tPartition %u to %u: %lu nodes.\n", partition_i, partition_j, partitions[partition_i]->transfer_info[partition_j]->num_to_gpu);
                printf("\tPartition %u from %u: %lu nodes.\n", partition_i, partition_j, partitions[partition_i]->transfer_info[partition_j]->num_from_gpu);
            }
        }
    }
}


/*
Second test version for assign_to_partitions, except doesn't make any assumptions regarding vertical partition rule. Just must make sure the which_partition function is accurate
*/
void assign_to_partitions_test_2(PartitionRule* partition_rule, Partition** partitions, unsigned int num_partitions, long num_nodes, double* center_coords, long num_edges, long* src_nodes, long* dst_nodes, int num_threads) {
    // Create mapping from global node id to which partition these nodes are being sent to
    // if there exists a node that should be sent to 2 different partitions, then throw an error

    // if index i is 3, then the ith node should be sent to partition 3. if index i is -1, then the ith node is a pure dst node
    int* node_to_partitions_inside = malloc(sizeof(int) * num_nodes);

    for (long node_i = 0; node_i < num_nodes; node_i++) {
        node_to_partitions_inside[node_i] = -1;
    }

    // iterate through all edges, if the src and dst partitions are different, then the src node should be sent to the dst partition.
    // in that case, mark node_to_partitions_inside[src] = dst partition.
    // if there's already a value of node_to_partitions_inside[src] that is not dst partition and also not -1, then throw an error (a single src node should not be able to go to different places) TODO: if you
    // want to move away from the vertical wall partitions (into something smarter like kd-tree style partitions), then you cannot assume that a single src node should only go to 1 dst partition

    #ifdef TIMING
        struct timespec t1 = get_time();
    #endif


    long src_node;
    long dst_node;

    // Calculate which edge indices each thread will operate over
    int quotient = num_edges / num_threads;
    int remainder = num_edges % num_threads;
    int start_ids[num_threads];
    int num_iters_array[num_threads];

    for (int i = 0; i < num_threads; i++) {
        int start_id = i * quotient + (i < remainder ? i : remainder);
        int num_iters = quotient + (i < remainder ? 1 : 0);

        start_ids[i] = start_id;
        num_iters_array[i] = num_iters;
    }

    #pragma omp parallel private(src_node, dst_node) num_threads(num_threads)
    {
        int thread_id = omp_get_thread_num();

        for (long edge_i = start_ids[thread_id]; edge_i < start_ids[thread_id] + num_iters_array[thread_id]; edge_i++) {
            src_node = src_nodes[edge_i];
            dst_node = dst_nodes[edge_i];

            double* src_coords_p = &center_coords[3 * src_node];
            double* dst_coords_p = &center_coords[3 * dst_node];

            int src_partition_id = (int) which_partition(partition_rule, src_coords_p);
            int dst_partition_id = (int) which_partition(partition_rule, dst_coords_p);

            if (src_partition_id != dst_partition_id) {

                if (node_to_partitions_inside[src_node] != -1 && node_to_partitions_inside[src_node] != dst_partition_id) {
                    printf("ERROR: global node %ld of partition %d has edges pointing to partitions %d and %d. We partitioned with vertical walls and thus assume that each src node should only go to 1 dst node. "
                            "If you are not using vertical wall "
                            "partitions, then you must change this assumption. The node's position is %lf %lf %lf\n"
                            "The dst node's positions is %lf %lf %lf\n", src_node, src_partition_id, dst_partition_id, node_to_partitions_inside[src_node], src_coords_p[0], src_coords_p[1], src_coords_p[2], dst_coords_p[0], dst_coords_p[1], dst_coords_p[2]);
                }

                node_to_partitions_inside[src_node] = dst_partition_id;
            }
        }
    }

    #ifdef TIMING
        struct timespec t2 = get_time();
        double elapsed = time_diff(t1, t2);
        printf("First loop within assign_to_partitions_test_2: %.9f\n", elapsed);

        t1 = get_time();
    #endif

    // Iterate through all nodes, if there exists an edge from that node to another partition, then update the necessary PartitionTransferInfo structs
    // otherwise, this node is a pure dst node and should be added to the necessary Partition struct
    int src_partition_id;
    double* src_coords;
    Partition* src_partition;

    int dst_partition_id;
    Partition* dst_partition;

    for (long node_i = 0; node_i < num_nodes; node_i++) {

        src_coords = &center_coords[3 * node_i];
        src_partition_id = (int) which_partition(partition_rule, src_coords);
        dst_partition_id = node_to_partitions_inside[node_i];

        src_partition = partitions[src_partition_id];
        dst_partition = partitions[dst_partition_id];

        if (dst_partition_id == -1) {

            src_partition = partitions[src_partition_id];

            src_partition->pure_dst_nodes[src_partition->num_pure_dst_nodes] = node_i;
            src_partition->num_pure_dst_nodes += 1;

        } else {
            // First update the "to" section of the src node's partition
            src_partition->transfer_info[dst_partition_id]->to_gpu[src_partition->transfer_info[dst_partition_id]->num_to_gpu] = node_i;
            src_partition->transfer_info[dst_partition_id]->num_to_gpu += 1;

            // Update the "from" section of the dst node's partition
            dst_partition->transfer_info[src_partition_id]->from_gpu[dst_partition->transfer_info[src_partition_id]->num_from_gpu] = node_i;
            dst_partition->transfer_info[src_partition_id]->num_from_gpu += 1;
        }
    }

    nodes_to_partition = node_to_partitions_inside;

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("Second loop in assign_to_partitions_test_2: %.9f\n", elapsed);
    #endif
}

/*
Returns the GPU id (0 indexed) that node_index belongs to. Assumes that partition_rule->walls coordinates are increasing order
Note: nodes directly on a boundary get assigned to the higher indexed partition. Each partition owns its left boundary. The highest partition owns both its left and right boundaries
*/
unsigned int which_partition(PartitionRule* partition_rule, double* coords) {

    for (unsigned int i = 0; i < partition_rule->num_walls; i++) {
        if (coords[partition_rule->dim_to_use] < partition_rule->walls[i]) {
            return i;
        }
    }

    // If we haven't returned yet, then our current node is beyond the final wall
    return partition_rule->num_walls;
}

/*
Initializes an empty partition object with memory allocated for the worst case scenario (minimizing mallocs/reallocs)
*/
void initialize_empty_partition(Partition* partition, unsigned int partition_id, long num_edges, long num_nodes, unsigned int num_partitions) {
    partition->gpu_id = partition_id;
    partition->num_edges = 0;

    // TODO: Currently allocating max amount of memory that these fields can take. Don't forget to free later.
    // May need to perform dynamic memory allocation if we run into memory issues.
    partition->edges_ids = malloc(sizeof(long) * num_edges);
    partition->num_pure_dst_nodes = 0;
    partition->pure_dst_nodes = malloc(sizeof(long) * num_nodes);

    partition->transfer_info = malloc(sizeof(PartitionTransferInfo*) * num_partitions);

    // Create and populate each PartitionTransferInfo structs and their corresponding fields
    for (unsigned int i = 0; i < num_partitions; i++) {
        if (i == partition_id) {
            // We don't need PartitionTransferInfo to/from our own partition!
            partition->transfer_info[i] = NULL;
        } else {
            partition->transfer_info[i] = malloc(sizeof(PartitionTransferInfo));

            partition->transfer_info[i]->other_gpu_id = i;

            partition->transfer_info[i]->num_from_gpu = 0;
            partition->transfer_info[i]->num_to_gpu = 0;

            partition->transfer_info[i]->to_gpu = malloc(sizeof(long) * num_nodes);
            partition->transfer_info[i]->from_gpu = malloc(sizeof(long) * num_nodes);

            partition->transfer_info[i]->edges_from_gpu = malloc(sizeof(BDirectedEdge*) * DEFAULT_ITEMS);
            partition->transfer_info[i]->edges_to_gpu = malloc(sizeof(BDirectedEdge*) * DEFAULT_ITEMS);

            partition->transfer_info[i]->num_edges_from_gpu = 0;
            partition->transfer_info[i]->num_edges_to_gpu = 0;
        }
    }

    partition->num_UDEs = 0;
    // NOTE: partition->pure_UDEs has not yet been allocated.
}

/*
Creates necessary partition metadata to be used in the which_partition function. Picks the dimension using cartesian coordinates. But creates walls using fractional coordinates
*/
void create_partition(PartitionRule* partition, double* center_coords, long num_nodes, unsigned int num_partitions, double* frac_coords) {
    // Find the id of the dimension with biggest different between max and min value using center_coords
    double max_0 = center_coords[0];
    double min_0 = center_coords[0];

    double max_1 = center_coords[1];
    double min_1 = center_coords[1];

    double max_2 = center_coords[2];
    double min_2 = center_coords[2];

    for (long i = 1; i < num_nodes; i++) {
        update_max_min(&center_coords[3 * i], &max_0, &min_0);
        update_max_min(&center_coords[3 * i + 1], &max_1, &min_1);
        update_max_min(&center_coords[3 * i + 2], &max_2, &min_2);
    }

    // TODO: testing, remove when done. Print all dim min and dim maxes
    PRINTF("Cartesian maxes: %f %f %f\n", max_0, max_1, max_2);
    PRINTF("Cartesian mins: %f %f %f\n", min_0, min_1, min_2);

    double diff_0 = max_0 - min_0;
    double diff_1 = max_1 - min_1;
    double diff_2 = max_2 - min_2;

    double diffs[] = {diff_0, diff_1, diff_2};


    unsigned int longest_dim = 0;
    double longest_dim_length = diffs[0];


    for (unsigned int i = 1; i < 3; i++) {
        if (diffs[i] > longest_dim_length) {
            longest_dim = i;
            longest_dim_length = diffs[i];
        }
    }

    // Create the partition struct + calculate wall location based on fractional coordinates
    double frac_min = frac_coords[longest_dim];
    double frac_max = frac_coords[longest_dim];
    double longest_dim_frac_length;

    // Find maxes and mins in fractional coordinates
    for (long i = 1; i < num_nodes; i++) {
        update_max_min(&frac_coords[3 * i + longest_dim], &frac_max, &frac_min);
    }
    PRINTF("Fractional longest dim max: %f\n", frac_max);
    PRINTF("Fractional longest dim min: %f\n", frac_min);

    longest_dim_frac_length = frac_max - frac_min;
    PRINTF("Longest dim length in fractional coordinates: %lf\n", longest_dim_frac_length);


    // Creating the Partition object
    partition->num_walls = num_partitions - 1;
    partition->dim_to_use = longest_dim;
    partition->max_val = frac_max;
    partition->min_val = frac_min;

    double* walls = malloc(sizeof(double) * partition->num_walls);

    for (unsigned int i = 1; i < num_partitions; i++) {
        walls[i - 1] = (i * (longest_dim_frac_length / num_partitions)) + EPSILON + partition->min_val;
    }

    partition->walls = walls;

    // Check if there are any atoms directly on the walls themselves. If this is the case, move the wall.
    bool has_collisions = true;

    while (has_collisions) {
        has_collisions = false;
        PRINTF("Checking collisions... \n");
        for (unsigned int wall_i = 0; wall_i < partition->num_walls; wall_i++) {
            for (long node_i = 0; node_i < num_nodes; node_i++) {
                if (frac_coords[3 * node_i + partition->dim_to_use] == partition->walls[wall_i]) {
                    has_collisions = true;
                    printf("Collision b/w atom and partition wall, moving wall.\n");

                    partition->walls[wall_i] += EPSILON;
                }
            }
        }
    }
}

void update_max_min(double* curr_val, double* max, double* min) {
    if (*curr_val > *max) {
        *max = *curr_val;
    }

    if (*curr_val < *min) {
        *min = *curr_val;
    }
}

/*
Gets minimum of a and b, if they are equal, returns a
*/
double min(double a, double b) {
    return (a < b) ? a : b;
}

/*
Gets maximum of a and b, if they are equal, returns a
*/
double max(double a, double b) {
    return (a > b) ? a : b;
}

/*
Gets the minimum of a and b if a and b are both longs
*/
long min_long(long a, long b) {
    return (a < b) ? a : b;
}


/*
Gets the  maximum of a and b if a and b are both longs
*/
long max_long(long a, long b) {
    return (a > b) ? a : b;
}

/* Returns a struct timespec with the current CPU Time */
struct timespec get_time() {
    struct timespec t1;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
    return t1;
}

/* Calculates the time elapsed between two timespec structs */
double time_diff(struct timespec t1, struct timespec t2) {
    return (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
}

/*
Checks the dimensions of the vertical wall partition rule and makes sure that the partitions aren't too small. If so, throws a warning message. Note that the lattice is a 3x3 matrix where the COLUMNS are lattice vectors
*/
int check_partition_size(unsigned int num_partitions, PartitionRule* partition_rule, double* lattice, double atom_cutoff, double bond_cutoff, bool use_bond_graph) {
    unsigned int dim_to_use = partition_rule->dim_to_use;
    double lattice_vector[] = {lattice[dim_to_use], lattice[dim_to_use + 3], lattice[dim_to_use + 6]};
    double vector_norm = sqrt(lattice_vector[0] * lattice_vector[0] + lattice_vector[1] * lattice_vector[1] + lattice_vector[2] * lattice_vector[2]);

    double partition_width = partition_rule->walls[0] * vector_norm;

    if (partition_width <= 2 * (atom_cutoff + bond_cutoff) && use_bond_graph) {
        printf("Bond graph is enabled but atom_cutoff is %f and bond_cutoff is %f. The partition width is %f which is <= 2 * (atom_cutoff + bond_cutoff) (%f).\nA wall width that is <= 2 * (atom_cutoff + bond_cutoff) will be inefficient.\n", atom_cutoff, bond_cutoff, partition_width, 2 * (atom_cutoff + bond_cutoff));
        printf("You should reduce the # of partitions. If you cannot fit your system on a reduced # of partitions, then your system is probably too dense\nNo vanilla distributed graph algorithm will be able to help you. Contact Kevin (kevinhan@cmu.edu) if you need help with this.\n");
        // printf("Going to fall back to non-distributed inference. But do try to fit your system on a reduced # of partitions.\n");
        return -1;
    } else if (partition_width <= 2 * atom_cutoff && !use_bond_graph) {
        printf("Bond graph is disabled and atom_cutoff is %f. Total wall width is %f which is <= 2 * (atom_cutoff) (%f).\nA wall width that is <= 2 * (atom_cutoff) will be inefficient.\n", atom_cutoff, partition_width, 2 * (atom_cutoff));
        printf("You should reduce the # of partitions. If you cannot fit your system on a reduced # of partitions, then your system is probably too dense\nNo vanilla distributed graph algorithm will be able to help you. Contact Kevin (kevinhan@cmu.edu) if you need help with this.\n");
        // printf("Going to fall back to non-distributed inference. But do try to fit your system on a reduced # of partitions.\n");
        return -1;
    }
    return 0;
}
