void intra_parallel_find_points_in_spheres_c(
    const double *all_coords, int all_coords_rows,
    const double *center_coords, int center_coords_rows,
    double r,
    const long *pbc,
    const double *lattice,
    long **out_index_1,
    long **out_index_2,
    double **out_offsets,
    double **out_distances,
    long **out_within_bond_r_indices,
    long *out_num_within_bond_r_indices,
    long *out_num_edges,
    double bond_r,
    double tol,
    int number_of_threads
);
