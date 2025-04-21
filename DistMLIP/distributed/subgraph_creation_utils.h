#include <stdbool.h>

#ifdef DEBUG
    #define PRINTF(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define PRINTF(fmt, ...) // Empty, does nothing when DEBUG is not set
#endif

extern const double EPSILON;
extern const double NUMERICAL_TOL;
extern const long DEFAULT_ITEMS;
extern const long LINE_GRAPH_DEFAULT_ITEMS;

typedef struct PartitionTransferInfo PartitionTransferInfo;

typedef struct _BDirectedEdge {
    long local_id; // id given to this BDE used when creating lines in line graph
    long global_id; // id given to this BDE at the time of its creation
    double* offset; //shape (3, )
    long init_src_node; // the src node of the DE in the atom graph
    long init_dst_node; // the dst "                                              "
    bool needs_in_line; // whether or not the current UDE needs an incoming line in bond graph (only border BDEs will have this set to False). Used to minimize unnecessary in-degrees for nodes within bond graph
} BDirectedEdge;

typedef struct _PartitionRuleVertical {
    unsigned int num_walls;
    unsigned int dim_to_use; // value must be 0, 1, or 2
    double* walls; // shape: (num_walls, )  scalar representing point along dim_to_use to partition space
    double min_val; // minimum value for all nodes along the dim_to_use dimension
    double max_val; // maximum value for all nodes along the dim_to_use dimension
} PartitionRule;

typedef struct _Partition {
    unsigned int gpu_id;
    long num_edges;
    long* edges_ids; // global indices to the edges in src_nodes and dst_nodes that are pure edges
    PartitionTransferInfo** transfer_info; // List of PartitionTransferInfo pointers representing data to transfer to/from various GPUs
                                          // The gpu_id index of this list will be the NULL pointer
    long* pure_dst_nodes; // nodes that don't get transferred to/from various GPUs
    long num_pure_dst_nodes;

    long num_UDEs; // number of UDEs assigned to this GPU
    BDirectedEdge** pure_UDEs; // array of pointers to UDE structs belonging purely to this partition
} Partition;



typedef struct PartitionTransferInfo {
    unsigned int other_gpu_id; // ID of the gpu/partition that we're keeping track of to/from data for
    long* to_gpu; // list of nodes that we send to other_gpu_id. this may hold duplicate information as the other_gpu_id's partition's from_gpu data. that's okay though
    long* from_gpu; // list of nodes that we receive from other_gpu_id
    long num_to_gpu; // # of nodes to_gpu
    long num_from_gpu; // # of nodes from_gpu

    BDirectedEdge** edges_to_gpu; // array of UDE pointers
    BDirectedEdge** edges_from_gpu; // array of UDE pointers

    long num_edges_to_gpu;
    long num_edges_from_gpu;
} PartitionTransferInfo;

typedef struct _AdjListEntry {
    BDirectedEdge** edges; // Array of pointers to undirected edges
    long num_edges;

} AdjListEntry;

typedef struct _AdjList {
    AdjListEntry* array_of_entries; // array of _AdjListEntry structs. for now, we determine which UDE goes into which entry in the table by indexing using the minimum node index
    long num_entries; // number of entries in the table, should be the same as number of nodes (under the current implementation of the table)
} AdjList;



typedef struct _Results {
    Partition** partitions;
    long** local_edge_src_nodes;
    long** local_edge_dst_nodes;
    long** global_id_arrays;
    long** global_id_markers;
    long* num_atoms_per_partition;
    double** local_center_coords;

    long** line_src_nodes;
    long** line_dst_nodes;
    long** center_atom_indices;
    long* line_num_edges;
    long* num_UDEs_per_partition;
    long** UDE_marker_arrays;

    long** local_bond_mapping_DE;
    long** local_bond_mapping_UDE;
    long* num_bond_mapping;

    long** G2L_DE_mappings; // TODO: for debugging purposes, remove when done. mappings from global edge id to local edge id per partition
} Results;

//TODO: large amounts of the code can be parallelized.
void create_partition(PartitionRule* partition, double* center_coords, long num_nodes, unsigned int num_partitions, double* frac_coords);
void update_max_min(double* curr_val, double* max, double* min);
void initialize_empty_partition(Partition* partition, unsigned int partition_id, long num_edges, long num_nodes, unsigned int num_partitions);
unsigned int which_partition(PartitionRule* partition_rule, double* coords);
void assign_to_partition_pure(PartitionRule* partition_rule, double min, double max, double* coord, Partition** partitions, long node_i, unsigned int partition_i);
void assign_to_partition_border(PartitionRule* partition_rule, double min, double max, double* coord, Partition** partitions, long node_i, unsigned int dst_part_id);
void assign_to_partitions_test_2(PartitionRule* partition_rule, Partition** partitions, unsigned int num_partitions, long num_nodes, double* center_coords, long num_edges, long* src_nodes, long* dst_nodes, int num_threads);
void print_partitions(Partition** partitions, unsigned int num_partitions);
void create_global_id_array(Partition* partition, long* global_id_array, long* global_id_marker, unsigned int num_partitions);
long get_total_num_atoms(Partition* partition, unsigned int num_partitions);
double min(double a, double b);
double max(double a, double b);
unsigned long hash_function(long x, long y);
long min_long(long a, long b);
//bool DE_UDE_equivalent(UndirectedEdge* UDE, long DE_src_node, long DE_dst_node, double* DE_offset);
long max_long(long a, long b);
void localize_UDE_and_create_UDE_markers(Partition* partition, unsigned int num_partitions, long* BDE_marker_array, long* G2L_BDE_mapping);
bool close_enough(double a, double b);
long get_num_UDEs_in_partition(Partition* partition, unsigned int num_partitions);
//void add_UDE_to_entry(AdjListEntry* entry, UndirectedEdge* to_add);
void add_BDE_to_entry(AdjListEntry* entry, BDirectedEdge* to_add);
struct timespec get_time();
double time_diff(struct timespec t1, struct timespec t2);
int check_partition_size(unsigned int num_partitions, PartitionRule* partition_rule, double* lattice, double atom_cutoff, double bond_cutoff, bool use_bond_graph);

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
    double* frac_coords, // fractional coordinates of each node: (num_nodes, 3)
    double* lattice // lattice matrix (3x3), c-contiguous, [:, ::1]
);
