#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include "subgraph_creation_utils.h"
#include "fpis.h"
#include <stdlib.h>
#include <omp.h>


void fractional_to_cartesian(
    const double* frac_coords,
    const double* lattice,
    double* cart_coords,
    int num_atoms
) {
    for (int i = 0; i < num_atoms; i++) {
        // Get fractional coordinates for atom `i`
        double u = frac_coords[3 * i];
        double v = frac_coords[3 * i + 1];
        double w = frac_coords[3 * i + 2];

        // Compute Cartesian coordinates
        cart_coords[3 * i]     = u * lattice[0] + v * lattice[1] + w * lattice[2]; // x
        cart_coords[3 * i + 1] = u * lattice[3] + v * lattice[4] + w * lattice[5]; // y
        cart_coords[3 * i + 2] = u * lattice[6] + v * lattice[7] + w * lattice[8]; // z
    }
}

Results* create_subgraphs(
    double* all_coords,
    int num_all_coords,
    double r,
    long* pbc,
    double* lattice,
    unsigned int num_partitions,
    double bond_r,
    double tol,
    int number_of_threads,
    bool use_bond_graph,
    double* frac_coords // TODO: eventually, calculate this from all_coords and lattice instead of relying on ase's method (ase's method is slower)
);

void free_allocated_memory(void* ptr) {
    free(ptr);
}

// Custom deallocator for the C buffer
void buffer_deallocator(PyObject *capsule) {
    void *buffer = PyCapsule_GetPointer(capsule, NULL);
    if (buffer) {
        free(buffer);  // Free the C buffer
    }
}

// Wraps the inputted buffer with a numpy array (assume buffer is c-contiguous) and transfers ownership to the np array via
// a capsule. If the np array is garbage collected, the underlying memory gets freed as well
PyObject* buf_to_np_array(void* buffer, npy_intp* dims, int num_dims, int typenum) {
    if (num_dims == 0 || !dims || (num_dims > 0 && dims[0] == 0)) {
        PyArray_Descr* descr = PyArray_DescrFromType(NPY_FLOAT);
        return PyArray_Empty(1, dims, descr, 0); // Return empty np array with 1 dimension of size 1
    }

    PyObject* array = PyArray_SimpleNewFromData(num_dims, dims, typenum, buffer);

    if (!array) {
        free(buffer);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy array.");
        return NULL;
    }

    // Attach a capsule to manage the memory
    PyObject* capsule = PyCapsule_New(buffer, NULL, buffer_deallocator);
    if (!capsule) {
        Py_DECREF(array);
        free(buffer);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create capsule for memory management.");
        return NULL;
    }

    // Set the capsule as the base object for the array
    if (PyArray_SetBaseObject((PyArrayObject*)array, capsule) < 0) {
        Py_DECREF(array);
        Py_DECREF(capsule);
        free(buffer);  // Free memory if setting the base object fails
        PyErr_SetString(PyExc_RuntimeError, "Failed to set capsule as base object.");
        return NULL;
    }

    return array;
}

static PyObject* get_subgraphs(PyObject *self, PyObject *args) {
    PyArrayObject *all_coords_double_npy, *pbc_long_npy, *lattice_double_npy, *frac_coords_double_npy;
    double r, bond_r, tol;
    int num_threads;
    unsigned int num_partitions;
    npy_bool use_bond_graph_npy;
    bool use_bond_graph;
    int num_nodes;

    // Parse the input arguments
    if (!PyArg_ParseTuple(args, "O!dO!O!IddibO!",
                          &PyArray_Type, &all_coords_double_npy,
                          &r,
                          &PyArray_Type, &pbc_long_npy,
                          &PyArray_Type, &lattice_double_npy,
                          &num_partitions,
                          &bond_r,
                          &tol,
                          &num_threads,
                          &use_bond_graph_npy,
                          &PyArray_Type, &frac_coords_double_npy
                        )) {
        printf("Failed to parse input arguments. Inputs should be of the following format: double[:, ::1] all_coords, double r, "
                "long[::1] pbc, const double[:, ::1] lattice, int num_nodes, unsigned int num_partitions, double bond_r=3.0, double tol=1e-8, "
                "int number_of_threads=256, bint use_bond_graph=True, frac_coords\n");
        return NULL; // Return NULL if argument parsing fails
    }


    // Validate input array types
    if (PyArray_TYPE(all_coords_double_npy) != NPY_DOUBLE || PyArray_NDIM(all_coords_double_npy) != 2) {
        PyErr_SetString(PyExc_TypeError, "all_coords array must be a 2D array of doubles (float64)");
        return NULL;
    }

    if (PyArray_TYPE(pbc_long_npy) != NPY_LONG || PyArray_NDIM(pbc_long_npy) != 1) {
        PyErr_SetString(PyExc_TypeError, "pbc array must be a 1D array of longs (int64)");
        return NULL;
    }

    if (PyArray_TYPE(lattice_double_npy) != NPY_DOUBLE || PyArray_NDIM(lattice_double_npy) != 2) {
        PyErr_SetString(PyExc_TypeError, "lattice array must be a 2D array of doubles (float64)");
        return NULL;
    }

    if (PyArray_TYPE(frac_coords_double_npy) != NPY_DOUBLE || PyArray_NDIM(frac_coords_double_npy) != 2) {
        PyErr_SetString(PyExc_TypeError, "fractional coordinates array must be a 2D array of doubles (float64)");
        return NULL; // TODO: it's significantly faster if this is calculated in c code instead of using the ase get_scaled_positions method
    }                // get_scaled_positions is still quite fast though, so this is not high priority

    num_nodes = (int) PyArray_SIZE(all_coords_double_npy) / 3;
    use_bond_graph = (bool) use_bond_graph_npy;

    double* all_coords = (double*) PyArray_DATA(all_coords_double_npy);
    long* pbc = (long*) PyArray_DATA(pbc_long_npy);
    double* lattice = (double*) PyArray_DATA(lattice_double_npy);
    double* frac_coords = (double*) PyArray_DATA(frac_coords_double_npy);


    // Run FPIS and then create_subgraphs -------------------------------------
    long* src_nodes;
    long* dst_nodes;
    long fpis_num_edges;
    double* offsets;
    double* distances;
    long* within_bond_r_indices;
    long num_within_bond_r_indices;
    // The above are all FPIS primitives

    #ifdef TIMING
        double elapsed;
        struct timespec t0, t1, t2, t3, t4;
        t0 = get_time();
    #endif
    intra_parallel_find_points_in_spheres_c(all_coords, num_nodes,
                                        all_coords, num_nodes,
                                        r, pbc, lattice, &src_nodes,
                                        &dst_nodes, &offsets, &distances,
                                        &within_bond_r_indices, &num_within_bond_r_indices,
                                        &fpis_num_edges, bond_r, tol, num_threads);

    #ifdef TIMING
        t1 = get_time();
        elapsed = time_diff(t0, t1);
        printf("TIMING: FPIS time: %lf\n", elapsed);
    #endif


    // Get wrapped cartesian coordinates using the (wrapped) fractional coordinates
    double* wrapped_cart_coords = malloc(sizeof(double) * (num_nodes * 3));
    fractional_to_cartesian(frac_coords, lattice, wrapped_cart_coords, num_nodes);

    #ifdef TIMING
        t2 = get_time();
        elapsed = time_diff(t1, t2);
        printf("TIMING: wrapped frac to wrapped cartesian timing: %f\n", elapsed);
        fflush(stdout);
    #endif
    Results* results;
    int result_msg = get_features(fpis_num_edges, src_nodes, dst_nodes,
                                    wrapped_cart_coords, num_nodes,
                                    num_partitions, r, distances,
                                    bond_r, num_within_bond_r_indices,
                                    within_bond_r_indices, offsets, num_threads,
                                    use_bond_graph, frac_coords, lattice, &results);

    #ifdef TIMING
        t3 = get_time();
        elapsed = time_diff(t2, t3);
        printf("TIMING: get_features time: %f\n", elapsed);
        fflush(stdout);
    #endif

    if (result_msg == -4) {
        PyErr_SetString(PyExc_RuntimeError, "Partition walls are too close together. See above message.");
        return NULL;
    } else if (result_msg == -3) {
        PyErr_SetString(PyExc_RuntimeError, "There contain self edges in atom graph, partition walls are too close.");
    } else if (result_msg == -2) {
        PyErr_SetString(PyExc_RuntimeError, "num_partitions must be >=2");
    }
    
    free(wrapped_cart_coords);
    // --------------------------------------------------------------------
    // Convert values in results to numpy arrays and python lists
    unsigned int partition_i;
    PyObject* num_UDEs_per_partition_pylist = PyList_New(num_partitions);

    // local_edge_src_nodes
    PyObject* local_edges_src_nodes_pylist = PyList_New(num_partitions);
    PyObject* local_edges_dst_nodes_pylist = PyList_New(num_partitions);
    PyObject* markers_pylist = PyList_New(num_partitions);
    PyObject* local_coords_pylist = PyList_New(num_partitions);
    PyObject* global_ids_pylist = PyList_New(num_partitions);
    PyObject* local_lines_src_nodes_pylist = PyList_New(num_partitions);
    PyObject* local_lines_dst_nodes_pylist = PyList_New(num_partitions);
    PyObject* line_markers_pylist = PyList_New(num_partitions);
    PyObject* local_bond_mapping_DE_pylist = PyList_New(num_partitions);
    PyObject* local_bond_mapping_UDE_pylist = PyList_New(num_partitions);
    PyObject* L2G_mapping_pylist = PyList_New(num_partitions);
    PyObject* local_center_atom_indices_pylist =  PyList_New(num_partitions);

    PyObject* G2L_mapping_pylist = PyList_New(num_partitions); // TODO: this should be removed once done testing

    for (partition_i = 0; partition_i < num_partitions; partition_i++) {
        npy_intp edges_dims[1] = {results->partitions[partition_i]->num_edges};
        npy_intp markers_dims[1] = {2 * num_partitions + 1};
        npy_intp atoms_dims[1] = {results->num_atoms_per_partition[partition_i]};
        npy_intp atoms_coords_dims[2] = {results->num_atoms_per_partition[partition_i], 3};

        PyObject* curr_local_edges_src_nodes = buf_to_np_array(results->local_edge_src_nodes[partition_i], edges_dims, 1, NPY_LONG);
        PyObject* curr_local_edges_dst_nodes = buf_to_np_array(results->local_edge_dst_nodes[partition_i], edges_dims, 1, NPY_LONG);

        PyList_SetItem(local_edges_src_nodes_pylist, partition_i, curr_local_edges_src_nodes);
        PyList_SetItem(local_edges_dst_nodes_pylist, partition_i, curr_local_edges_dst_nodes);

        PyObject* curr_markers = buf_to_np_array(results->global_id_markers[partition_i], markers_dims, 1, NPY_LONG);
        PyList_SetItem(markers_pylist, partition_i, curr_markers);

        PyObject* curr_local_coords = buf_to_np_array(results->local_center_coords[partition_i], atoms_coords_dims, 2, NPY_DOUBLE);
        PyList_SetItem(local_coords_pylist, partition_i, curr_local_coords);

        PyObject* curr_global_ids = buf_to_np_array(results->global_id_arrays[partition_i], atoms_dims, 1, NPY_LONG);
        PyList_SetItem(global_ids_pylist, partition_i, curr_global_ids);


        PyObject* curr_L2G_mapping = buf_to_np_array(results->partitions[partition_i]->edges_ids, edges_dims, 1, NPY_LONG);
        PyList_SetItem(L2G_mapping_pylist, partition_i, curr_L2G_mapping);

        if (use_bond_graph) {
            PyObject* curr_G2L_mapping = buf_to_np_array(results->G2L_DE_mappings[partition_i], edges_dims, 1, NPY_LONG);
            PyList_SetItem(G2L_mapping_pylist, partition_i, curr_G2L_mapping);

            PyObject* curr_line_markers = buf_to_np_array(results->UDE_marker_arrays[partition_i], markers_dims, 1, NPY_LONG);
            PyList_SetItem(line_markers_pylist, partition_i, curr_line_markers);

            PyList_SetItem(num_UDEs_per_partition_pylist, partition_i, PyLong_FromLong(results->num_UDEs_per_partition[partition_i]));

            npy_intp lines_dims[1] = {results->line_num_edges[partition_i]};
            PyObject* curr_center_atom_indices = buf_to_np_array(results->center_atom_indices[partition_i], lines_dims, 1, NPY_LONG);
            PyList_SetItem(local_center_atom_indices_pylist, partition_i, curr_center_atom_indices);

            npy_intp mapping_dims[1] = {results->num_bond_mapping[partition_i]};
            PyObject* curr_bond_mapping_DE = buf_to_np_array(results->local_bond_mapping_DE[partition_i], mapping_dims, 1, NPY_LONG);
            PyObject* curr_bond_mapping_UDE = buf_to_np_array(results->local_bond_mapping_UDE[partition_i], mapping_dims, 1, NPY_LONG);

            PyList_SetItem(local_bond_mapping_DE_pylist, partition_i, curr_bond_mapping_DE);
            PyList_SetItem(local_bond_mapping_UDE_pylist, partition_i, curr_bond_mapping_UDE);

            PyObject* curr_local_lines_src_nodes = buf_to_np_array(results->line_src_nodes[partition_i], lines_dims, 1, NPY_LONG);
            PyObject* curr_local_lines_dst_nodes = buf_to_np_array(results->line_dst_nodes[partition_i], lines_dims, 1, NPY_LONG);

            PyList_SetItem(local_lines_src_nodes_pylist, partition_i, curr_local_lines_src_nodes);
            PyList_SetItem(local_lines_dst_nodes_pylist, partition_i, curr_local_lines_dst_nodes);

        }

    }


    npy_intp fpis_edges_dims[1] = {fpis_num_edges};
    npy_intp fpis_offsets_dims[2] = {fpis_num_edges, 3};
    npy_intp fpis_within_bond_r_dims[1] = {num_within_bond_r_indices};
    PyObject* fpis_src_nodes = buf_to_np_array(src_nodes, fpis_edges_dims, 1, NPY_LONG);
    PyObject* fpis_dst_nodes = buf_to_np_array(dst_nodes, fpis_edges_dims, 1, NPY_LONG);
    PyObject* fpis_offsets = buf_to_np_array(offsets, fpis_offsets_dims, 2, NPY_DOUBLE);
    PyObject* fpis_dists = buf_to_np_array(distances, fpis_edges_dims, 1, NPY_DOUBLE);
    PyObject* fpis_within_bond_r_indices = buf_to_np_array(within_bond_r_indices, fpis_within_bond_r_dims, 1, NPY_LONG);

    // Free unnecessary memory now. Remember that a lot of memory gets transferred over to numpy (and thus shouldn't be freed)

    for (partition_i = 0; partition_i < num_partitions; partition_i++) {
        // free(results->global_id_arrays[partition_i]);
        // free(results->global_id_markers[partition_i]);

        // free(results->partitions[partition_i]->edges_ids);

        free(results->partitions[partition_i]->pure_dst_nodes);

        if (use_bond_graph) {
            free(results->partitions[partition_i]->pure_UDEs);
        }

        // free(results->local_edge_src_nodes[partition_i]);
        // free(results->local_edge_dst_nodes[partition_i]);

        // free(results->local_center_coords[partition_i]);

        if (use_bond_graph) {
            // free(results->line_src_nodes[partition_i]);
            // free(results->line_dst_nodes[partition_i]);

            // free(results->center_atom_indices[partition_i]);

            // free(results->UDE_marker_arrays[partition_i]);

            // free(results->local_bond_mapping_DE[partition_i]);
            // free(results->local_bond_mapping_UDE[partition_i]);
        }

        // Free transfer info from each partition struct
        for (unsigned int partition_j = 0; partition_j < num_partitions; partition_j++) {
            if (partition_i == partition_j) {
                continue;
            } else {
                free(results->partitions[partition_i]->transfer_info[partition_j]->to_gpu);
                free(results->partitions[partition_i]->transfer_info[partition_j]->from_gpu);

                if (use_bond_graph) {
                    free(results->partitions[partition_i]->transfer_info[partition_j]->edges_from_gpu);
                    free(results->partitions[partition_i]->transfer_info[partition_j]->edges_to_gpu);
                }

                free(results->partitions[partition_i]->transfer_info[partition_j]);
            }
        }

        free(results->partitions[partition_i]->transfer_info);
        free(results->partitions[partition_i]);

        if (use_bond_graph) {
            // TODO: transfer this to subgraph_creation_utils when done testing
            // free(results->G2L_DE_mappings[partition_i]);
        }

    }
    free(results->global_id_arrays);
    free(results->global_id_markers);

    free(results->partitions);
    free(results->local_edge_src_nodes);
    free(results->local_edge_dst_nodes);

    free(results->num_atoms_per_partition);

    free(results->local_center_coords);

    if (use_bond_graph) {
        free(results->line_src_nodes);
        free(results->line_dst_nodes);

        free(results->center_atom_indices);

        free(results->line_num_edges);
        free(results->num_UDEs_per_partition);

        free(results->UDE_marker_arrays);

        free(results->local_bond_mapping_DE);
        free(results->local_bond_mapping_UDE);

        free(results->num_bond_mapping);
    }


    if (use_bond_graph) {
        // TODO: transfer this to subgraph_creation_utils when done testing
        free(results->G2L_DE_mappings);
    }


    free(results);

    // Free memory allocated with pneighbors.pyx
    // free(src_nodes);
    // free(dst_nodes);
    // free(offsets);
    // free(distances);
    // free(within_bond_r_indices);

    // Create an empty tuple of size 19
    PyObject* ret = PyTuple_New(19);
    PyTuple_SET_ITEM(ret, 0, local_edges_src_nodes_pylist);
    PyTuple_SET_ITEM(ret, 1, local_edges_dst_nodes_pylist);
    PyTuple_SET_ITEM(ret, 2, markers_pylist);
    PyTuple_SET_ITEM(ret, 3, local_coords_pylist);
    PyTuple_SET_ITEM(ret, 4, global_ids_pylist);
    PyTuple_SET_ITEM(ret, 5, fpis_src_nodes);
    PyTuple_SET_ITEM(ret, 6, fpis_dst_nodes);
    PyTuple_SET_ITEM(ret, 7, fpis_offsets);
    PyTuple_SET_ITEM(ret, 8, fpis_dists);
    PyTuple_SET_ITEM(ret, 9, local_lines_src_nodes_pylist);
    PyTuple_SET_ITEM(ret, 10, local_lines_dst_nodes_pylist);
    PyTuple_SET_ITEM(ret, 11, fpis_within_bond_r_indices);
    PyTuple_SET_ITEM(ret, 12, line_markers_pylist);
    PyTuple_SET_ITEM(ret, 13, num_UDEs_per_partition_pylist);
    PyTuple_SET_ITEM(ret, 14, local_bond_mapping_DE_pylist);
    PyTuple_SET_ITEM(ret, 15, local_bond_mapping_UDE_pylist);
    PyTuple_SET_ITEM(ret, 16, L2G_mapping_pylist);
    PyTuple_SET_ITEM(ret, 17, G2L_mapping_pylist);
    PyTuple_SET_ITEM(ret, 18, local_center_atom_indices_pylist);

    #ifdef TIMING
        t4 = get_time();
        elapsed = time_diff(t3, t4);
        printf("TIMING: setting and freeing in subgraph_creation_fast: %f\n", elapsed);
        fflush(stdout);
    #endif

    return ret;


    return PyTuple_Pack(19, local_edges_src_nodes_pylist,
                            local_edges_dst_nodes_pylist,
                            markers_pylist,
                            local_coords_pylist,
                            global_ids_pylist,
                            fpis_src_nodes,
                            fpis_dst_nodes,
                            fpis_offsets,
                            fpis_dists,
                            local_lines_src_nodes_pylist,
                            local_lines_dst_nodes_pylist,
                            fpis_within_bond_r_indices,
                            line_markers_pylist,
                            num_UDEs_per_partition_pylist,
                            local_bond_mapping_DE_pylist,
                            local_bond_mapping_UDE_pylist,
                            L2G_mapping_pylist,
                            G2L_mapping_pylist,
                            local_center_atom_indices_pylist);
}

// Define module methods
static PyMethodDef SubgraphsMethods[] = {
    {"get_subgraphs_fast", get_subgraphs, METH_VARARGS, "Create subgraphs for distributed inference"},
    {NULL, NULL, 0, NULL}
};

// Define the module
static struct PyModuleDef SubgraphCreationModule = {
    PyModuleDef_HEAD_INIT,
    "subgraph_creation_fast",
    "A fast module for subgraph creation implemented purely in c.",
    -1,
    SubgraphsMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_subgraph_creation_fast(void) {
    import_array(); // Initialize NumPy API
    return PyModule_Create(&SubgraphCreationModule);
}
