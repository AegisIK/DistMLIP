#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <time.h>

#define M_PI 3.14159265358979323846
#define MALLOC_MULTIPLIER 10

void *safe_malloc(size_t size) {
    void *ptr;
    if (size == 0) {
        return NULL;
    }
    ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Memory allocation of %zu bytes failed!\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void *safe_realloc(void *ptr_orig, size_t size) {
    void *ptr;
    if (size == 0) {
        return NULL;
    }
    ptr = realloc(ptr_orig, size);
    if (ptr == NULL) {
        fprintf(stderr, "Memory reallocation of %zu bytes failed!\n", size);
        exit(EXIT_FAILURE);
    }
    return ptr;
}

void get_cube_neighbors(long ncube[3], long **neighbor_map, long nb_cubes);
int compute_offset_vectors(long **ovectors, long n);
double distance2(const double *m1, const double *m2, long index1, long index2, long size);
void get_bounds(const double *frac_coords, int frac_coords_rows, double maxr[3], const long pbc[3], long max_bounds[3], long min_bounds[3]);
void get_frac_coords(const double *lattice, double *inv_lattice, const double *cart_coords, int coords_rows, double *frac_coords);
void matmul(const double *m1, int m1_rows, int m1_cols, const double *m2, int m2_rows, int m2_cols, double *out);
void matrix_inv(const double *matrix, double *inv);
double matrix_det(const double *matrix);
void get_max_r(const double *reciprocal_lattice, double maxr[3], double r);
void get_reciprocal_lattice(const double *lattice, double *reciprocal_lattice);
void recip_component(const double *a1, const double *a2, const double *a3, double *out);
double inner(const double *x, const double *y);
void cross(const double *x, const double *y, double *out);
double vec_norm(const double *vec, int size);
void max_and_min(const double *coords, int coords_rows, int coords_cols, double max_coords[3], double min_coords[3]);
void compute_cube_index(const double *coords, int coords_rows, const double global_min[3], double radius, long *return_indices);
void three_to_one(const long *label3d, int n, long ny, long nz, long *label1d);
int distance_vertices(const double center[8][3], const double off[8][3], double r);
void offset_cube(const double center[8][3], long n, long m, long l, double offsetted[8][3]);

// Implement other functions similarly
void get_cube_neighbors(long ncube[3], long **neighbor_map, long nb_cubes) {
    int i, j, k;
    int count = 0;
    long *counts = (long *)safe_malloc(nb_cubes * sizeof(long));
    long *cube_indices_3d = (long *)safe_malloc(nb_cubes * 3 * sizeof(long));
    long *cube_indices_1d = (long *)safe_malloc(nb_cubes * sizeof(long));

    long n = 1;
    long *ovectors;
    int n_ovectors = compute_offset_vectors(&ovectors, n);

    for (i = 0; i < nb_cubes; i++) {
        counts[i] = 0;
    }

    // Generate cube indices
    count = 0;
    for (i = 0; i < ncube[0]; i++) {
        for (j = 0; j < ncube[1]; j++) {
            for (k = 0; k < ncube[2]; k++) {
                cube_indices_3d[3*count] = i;
                cube_indices_3d[3*count+1] = j;
                cube_indices_3d[3*count+2] = k;
                count += 1;
            }
        }
    }

    // Convert 3D indices to 1D
    three_to_one(cube_indices_3d, nb_cubes, ncube[1], ncube[2], cube_indices_1d);

    for (i = 0; i < nb_cubes; i++) {
        for (j = 0; j < n_ovectors; j++) {
            long index3[3];
            index3[0] = ovectors[3*j] + cube_indices_3d[3*i];
            index3[1] = ovectors[3*j+1] + cube_indices_3d[3*i+1];
            index3[2] = ovectors[3*j+2] + cube_indices_3d[3*i+2];
            if ((index3[0] < ncube[0]) && (index3[0] >= 0) &&
                (index3[1] < ncube[1]) && (index3[1] >= 0) &&
                (index3[2] < ncube[2]) && (index3[2] >= 0)) {
                long index1;
                index1 = index3[0] * ncube[1] * ncube[2] + index3[1] * ncube[2] + index3[2];
                neighbor_map[i][counts[i]] = index1;
                counts[i] += 1;
            }
        }
    }

    free(cube_indices_3d);
    free(cube_indices_1d);
    free(counts);
    free(ovectors);
}

int compute_offset_vectors(long **ovectors, long n) {
    int i, j, k, ind;
    int count = 0;
    double center[8][3];
    double offset[8][3];

    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            for (k = 0; k < 2; k++) {
                ind = i * 4 + j * 2 + k;
                center[ind][0] = i - 0.5;
                center[ind][1] = j - 0.5;
                center[ind][2] = k - 0.5;
            }
        }
    }

    long ntotal = (2 * n + 1) * (2 * n + 1) * (2 * n + 1);
    *ovectors = (long *)safe_malloc(ntotal * 3 * sizeof(long));

    for (i = -n; i <= n; i++) {
        for (j = -n; j <= n; j++) {
            for (k = -n; k <= n; k++) {
                offset_cube(center, i, j, k, offset);
                if (distance_vertices(center, offset, n)) {
                    (*ovectors)[3*count] = i;
                    (*ovectors)[3*count+1] = j;
                    (*ovectors)[3*count+2] = k;
                    count += 1;
                }
            }
        }
    }

    *ovectors = (long *)safe_realloc(*ovectors, count * 3 * sizeof(long));
    return count;
}



// Function to compute the squared distance between two points
double distance2(
    const double *m1,
    const double *m2,
    long index1,
    long index2,
    long size
) {
    double s = 0.0;
    long i;
    for (i = 0; i < size; i++) {
        double diff = m1[index1 * size + i] - m2[index2 * size + i];
        s += diff * diff;
    }
    return s;
}

// Function to compute the bounds for translations
void get_bounds(
    const double *frac_coords, int frac_coords_rows,
    double maxr[3],
    const long pbc[3],
    long max_bounds[3],
    long min_bounds[3]
) {
    double max_fcoords[3];
    double min_fcoords[3];
    int i;

    // Compute the min and max of frac_coords
    max_and_min(frac_coords, frac_coords_rows, 3, max_fcoords, min_fcoords);

    for (i = 0; i < 3; i++) {
        min_bounds[i] = 0;
        max_bounds[i] = 1;
    }

    for (i = 0; i < 3; i++) {
        if (pbc[i]) {
            min_bounds[i] = (long)floor(min_fcoords[i] - maxr[i] - 1e-8);
            max_bounds[i] = (long)ceil(max_fcoords[i] + maxr[i] + 1e-8);
        }
    }
}

// Function to compute fractional coordinates
void get_frac_coords(
    const double *lattice,
    double *inv_lattice,
    const double *cart_coords,
    int coords_rows,
    double *frac_coords
) {
    matrix_inv(lattice, inv_lattice);
    matmul(cart_coords, coords_rows, 3, inv_lattice, 3, 3, frac_coords);
}

// Matrix multiplication function
void matmul(
    const double *m1, int m1_rows, int m1_cols,
    const double *m2, int m2_rows, int m2_cols,
    double *out
) {
    int i, j, k;
    int m = m1_rows;
    int n = m1_cols;
    int l = m2_cols;

    for (i = 0; i < m; i++) {
        for (j = 0; j < l; j++) {
            out[i * l + j] = 0.0;
            for (k = 0; k < n; k++) {
                out[i * l + j] += m1[i * n + k] * m2[k * l + j];
            }
        }
    }
}

// Function to compute the inverse of a 3x3 matrix
void matrix_inv(const double *matrix, double *inv) {
    double det = matrix_det(matrix);
    int i, j;

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            int a = (j + 1) % 3;
            int b = (i + 1) % 3;
            int c = (j + 2) % 3;
            int d = (i + 2) % 3;

            inv[i * 3 + j] = (matrix[a * 3 + b] * matrix[c * 3 + d] - matrix[c * 3 + b] * matrix[a * 3 + d]) / det;
        }
    }
}

// Function to compute the determinant of a 3x3 matrix
double matrix_det(const double *matrix) {
    return
        matrix[0 * 3 + 0] * (matrix[1 * 3 + 1] * matrix[2 * 3 + 2] - matrix[1 * 3 + 2] * matrix[2 * 3 + 1]) +
        matrix[0 * 3 + 1] * (matrix[1 * 3 + 2] * matrix[2 * 3 + 0] - matrix[1 * 3 + 0] * matrix[2 * 3 + 2]) +
        matrix[0 * 3 + 2] * (matrix[1 * 3 + 0] * matrix[2 * 3 + 1] - matrix[1 * 3 + 1] * matrix[2 * 3 + 0]);
}

// Function to compute the maximum repetitions in each direction
void get_max_r(const double *reciprocal_lattice, double maxr[3], double r) {
    int i;
    double recp_len;
    for (i = 0; i < 3; i++) {
        recp_len = vec_norm(&reciprocal_lattice[i * 3], 3);
        maxr[i] = ceil((r + 0.15) * recp_len / (2 * M_PI));
    }
}

// Function to compute the reciprocal lattice
void get_reciprocal_lattice(const double *lattice, double *reciprocal_lattice) {
    int i;
    for (i = 0; i < 3; i++) {
        const double *a1 = &lattice[i * 3];
        const double *a2 = &lattice[((i + 1) % 3) * 3];
        const double *a3 = &lattice[((i + 2) % 3) * 3];
        double *out = &reciprocal_lattice[i * 3];
        recip_component(a1, a2, a3, out);
    }
}

// Function to compute a reciprocal lattice vector
void recip_component(const double *a1, const double *a2, const double *a3, double *out) {
    double ai_cross_aj[3];
    double prod;
    cross(a2, a3, ai_cross_aj);
    prod = inner(a1, ai_cross_aj);
    for (int i = 0; i < 3; i++) {
        out[i] = 2 * M_PI * ai_cross_aj[i] / prod;
    }
}

// Function to compute the inner product of two 3D vectors
double inner(const double *x, const double *y) {
    double sum = 0.0;
    int i;
    for (i = 0; i < 3; i++) {
        sum += x[i] * y[i];
    }
    return sum;
}

// Function to compute the cross product of two 3D vectors
void cross(const double *x, const double *y, double *out) {
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
}

// Function to compute the norm (magnitude) of a vector
double vec_norm(const double *vec, int size) {
    int i;
    double sum = 0.0;
    for (i = 0; i < size; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrt(sum);
}

// Function to compute the min and max of coordinates
void max_and_min(
    const double *coords,
    int coords_rows,
    int coords_cols,
    double max_coords[3],
    double min_coords[3]
) {
    int i, j;
    for (i = 0; i < coords_cols; i++) {
        max_coords[i] = coords[i];
        min_coords[i] = coords[i];
    }
    for (i = 0; i < coords_rows; i++) {
        for (j = 0; j < coords_cols; j++) {
            double val = coords[i * coords_cols + j];
            if (val > max_coords[j]) {
                max_coords[j] = val;
            }
            if (val < min_coords[j]) {
                min_coords[j] = val;
            }
        }
    }
}

// Function to compute cube indices for given coordinates
void compute_cube_index(
    const double *coords,
    int coords_rows,
    const double global_min[3],
    double radius,
    long *return_indices
) {
    int i, j;
    for (i = 0; i < coords_rows; i++) {
        for (j = 0; j < 3; j++) {
            return_indices[i * 3 + j] = (long)(
                floor((coords[i * 3 + j] - global_min[j] + 1e-8) / radius)
            );
        }
    }
}

// Function to convert 3D indices to 1D indices
void three_to_one(
    const long *label3d,
    int n,
    long ny,
    long nz,
    long *label1d
) {
    int i;
    for (i = 0; i < n; i++) {
        label1d[i] = label3d[i * 3 + 0] * ny * nz + label3d[i * 3 + 1] * nz + label3d[i * 3 + 2];
    }
}

// Function to determine if any vertices are within a distance r
int distance_vertices(
    const double center[8][3],
    const double off[8][3],
    double r
) {
    int i, j;
    double d2;
    double r2 = r * r;
    for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++) {
            d2 = (center[i][0] - off[j][0]) * (center[i][0] - off[j][0]) +
                 (center[i][1] - off[j][1]) * (center[i][1] - off[j][1]) +
                 (center[i][2] - off[j][2]) * (center[i][2] - off[j][2]);
            if (d2 <= r2) {
                return 1;
            }
        }
    }
    return 0;
}

// Function to offset cube vertices
void offset_cube(
    const double center[8][3],
    long n, long m, long l,
    double offsetted[8][3]
) {
    int i, j, k, ind;
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 2; j++) {
            for (k = 0; k < 2; k++) {
                ind = i * 4 + j * 2 + k;
                offsetted[ind][0] = center[ind][0] + n;
                offsetted[ind][1] = center[ind][1] + m;
                offsetted[ind][2] = center[ind][2] + l;
            }
        }
    }
}

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
) {
    if (bond_r > r) {
        fprintf(stderr, "bond_r cannot be greater than regular cutoff\n");
        exit(EXIT_FAILURE);
    }

    int i, j, k, l, m;
    double maxr[3];
    double valid_min[3];
    double valid_max[3];
    double ledge;

    int n_center = center_coords_rows;
    int n_total = all_coords_rows;
    long nlattice = 1;

    long max_bounds[3] = {1, 1, 1};
    long min_bounds[3] = {0, 0, 0};

    double *frac_coords = (double *)safe_malloc(n_center * 3 * sizeof(double));
    double *all_fcoords = (double *)safe_malloc(n_total * 3 * sizeof(double));
    double *coords_in_cell = (double *)safe_malloc(n_total * 3 * sizeof(double));
    double *offset_correction = (double *)safe_malloc(n_total * 3 * sizeof(double));

    double inv_lattice[9];
    double reciprocal_lattice[9];

    int count = 0;
    int natoms = n_total;
    long ncube[3];
    double coord_temp[3];

    long *center_indices3 = (long *)safe_malloc(n_center * 3 * sizeof(long));
    long *center_indices1 = (long *)safe_malloc(n_center * sizeof(long));

    int failed_malloc = 0;
    long cube_index_temp;
    long link_index;
    double d_temp2;
    double r2 = r * r;
    double bond_r2 = bond_r * bond_r;

    if (r < 0.1) {
        ledge = 0.1;
    } else {
        ledge = r;
    }

    // Compute valid_min and valid_max
    max_and_min(center_coords, n_center, 3, valid_max, valid_min);
    for (i = 0; i < 3; i++) {
        valid_max[i] = valid_max[i] + r + tol;
        valid_min[i] = valid_min[i] - r - tol;
    }

    // Compute fractional coordinates
    get_frac_coords(lattice, inv_lattice, all_coords, n_total, offset_correction);
    for (i = 0; i < n_total; i++) {
        for (j = 0; j < 3; j++) {
            if (pbc[j]) {
                // Wrap atoms when this dimension is PBC
                all_fcoords[3*i + j] = fmod(offset_correction[3*i + j], 1.0);
                if (all_fcoords[3*i + j] < 0) {
                    all_fcoords[3*i + j] += 1.0;
                }
                offset_correction[3*i + j] = offset_correction[3*i + j] - all_fcoords[3*i + j];
            } else {
                all_fcoords[3*i + j] = offset_correction[3*i + j];
                offset_correction[3*i + j] = 0;
            }
        }
    }

    // Compute the reciprocal lattice
    get_reciprocal_lattice(lattice, reciprocal_lattice);
    get_max_r(reciprocal_lattice, maxr, r);

    // Get fractional coordinates of center points
    get_frac_coords(lattice, inv_lattice, center_coords, n_center, frac_coords);
    get_bounds(frac_coords, n_center, maxr, pbc, max_bounds, min_bounds);

    for (i = 0; i < 3; i++) {
        nlattice *= (max_bounds[i] - min_bounds[i]);
    }
    matmul(all_fcoords, n_total, 3, lattice, 3, 3, coords_in_cell);

    long max_count = 0;

    #ifdef TIMING
        double t1, t2, t3, t4, t5, t6;
        double elapsed;
        t1 = omp_get_wtime();
    #endif

    // First loop to calculate memory usage
    #pragma omp parallel for reduction(+:max_count) private(i, j, k, l, m, coord_temp) num_threads(number_of_threads)
    for (i = min_bounds[0]; i < max_bounds[0]; i++) {
        for (j = min_bounds[1]; j < max_bounds[1]; j++) {
            for (k = min_bounds[2]; k < max_bounds[2]; k++) {
                for (l = 0; l < n_total; l++) {
                    for (m = 0; m < 3; m++) {
                        coord_temp[m] = (double)i * lattice[m] +
                                        (double)j * lattice[3 + m] +
                                        (double)k * lattice[6 + m] +
                                        coords_in_cell[3*l + m];
                    }
                    if ((coord_temp[0] > valid_min[0]) &&
                        (coord_temp[0] < valid_max[0]) &&
                        (coord_temp[1] > valid_min[1]) &&
                        (coord_temp[1] < valid_max[1]) &&
                        (coord_temp[2] > valid_min[2]) &&
                        (coord_temp[2] < valid_max[2])) {
                        max_count += 1;
                    }
                }
            }
        }
    }

    #ifdef TIMING
        t2 = omp_get_wtime();
        elapsed = t2 - t1;
        printf("First loop elapsed: %.9f seconds\n", elapsed);
    #endif

    double *offsets_p = (double *)safe_malloc(max_count * 3 * sizeof(double));
    double *expanded_coords_p = (double *)safe_malloc(max_count * 3 * sizeof(double));
    long *indices_p = (long *)safe_malloc(max_count * sizeof(long));

    // Get translated images, coordinates and indices
    count = 0;
    #pragma omp parallel for private(i, j, k, l, m, coord_temp) collapse(4) num_threads(number_of_threads)
    for (i = min_bounds[0]; i < max_bounds[0]; i++) {
        for (j = min_bounds[1]; j < max_bounds[1]; j++) {
            for (k = min_bounds[2]; k < max_bounds[2]; k++) {
                for (l = 0; l < n_total; l++) {
                    for (m = 0; m < 3; m++) {
                        coord_temp[m] = (double)i * lattice[m] +
                                        (double)j * lattice[3 + m] +
                                        (double)k * lattice[6 + m] +
                                        coords_in_cell[3*l + m];
                    }
                    if ((coord_temp[0] > valid_min[0]) &&
                        (coord_temp[0] < valid_max[0]) &&
                        (coord_temp[1] > valid_min[1]) &&
                        (coord_temp[1] < valid_max[1]) &&
                        (coord_temp[2] > valid_min[2]) &&
                        (coord_temp[2] < valid_max[2])) {

                        int idx = 0;
                        #pragma omp atomic capture
                        idx = count++;

                        offsets_p[3*idx] = i;
                        offsets_p[3*idx+1] = j;
                        offsets_p[3*idx+2] = k;
                        indices_p[idx] = l;
                        expanded_coords_p[3*idx] = coord_temp[0];
                        expanded_coords_p[3*idx+1] = coord_temp[1];
                        expanded_coords_p[3*idx+2] = coord_temp[2];
                    }
                }
            }
        }
    }

    #ifdef TIMING
        t3 = omp_get_wtime();
        elapsed = t3 - t2;
        printf("Second loop elapsed: %.9f seconds\n", elapsed);
    #endif

    if (count == 0) {
        free(frac_coords);
        free(all_fcoords);
        free(coords_in_cell);
        free(offset_correction);
        free(center_indices1);
        free(center_indices3);

        free(offsets_p);
        free(expanded_coords_p);
        free(indices_p);
        fprintf(stderr, "No neighbors were found!\n");
        exit(EXIT_FAILURE);
    }

    natoms = count;

    // Construct linked cell list
    long *all_indices3 = (long *)safe_malloc(natoms * 3 * sizeof(long));
    long *all_indices1 = (long *)safe_malloc(natoms * sizeof(long));

    // Compute cube dimensions
    for (i = 0; i < 3; i++) {
        ncube[i] = (long)ceil((valid_max[i] - valid_min[i]) / ledge);
    }

    double *offsets = offsets_p;
    double *expanded_coords = expanded_coords_p;
    long *indices = indices_p;

    // Compute cube indices
    compute_cube_index(expanded_coords, natoms, valid_min, ledge, all_indices3);

    three_to_one(all_indices3, natoms, ncube[1], ncube[2], all_indices1);

    long nb_cubes = ncube[0] * ncube[1] * ncube[2];
    long *head = (long *)safe_malloc(nb_cubes * sizeof(long));
    long *atom_indices = (long *)safe_malloc(natoms * sizeof(long));
    memset(head, -1, nb_cubes * sizeof(long));
    memset(atom_indices, -1, natoms * sizeof(long));

    long **neighbor_map = (long **)safe_malloc(nb_cubes * sizeof(long *));
    for (i = 0; i < nb_cubes; i++) {
        neighbor_map[i] = (long *)safe_malloc(27 * sizeof(long));
        for (j = 0; j < 27; j++) {
            neighbor_map[i][j] = -1;
        }
    }

    get_cube_neighbors(ncube, neighbor_map, nb_cubes);
    for (i = 0; i < natoms; i++) {
        atom_indices[i] = head[all_indices1[i]];
        head[all_indices1[i]] = i;
    }

    // Get center atoms' cube indices
    compute_cube_index(center_coords, n_center, valid_min, ledge, center_indices3);
    three_to_one(center_indices3, n_center, ncube[1], ncube[2], center_indices1);

    max_count = 0;

    #ifdef TIMING
        t4 = omp_get_wtime();
    #endif

    // Third loop -- allocate ranges for each thread
    int quotient = n_center / number_of_threads;
    int remainder = n_center % number_of_threads;
    int start_ids[number_of_threads];
    int num_iters_array[number_of_threads];

    for (i = 0; i < number_of_threads; i++) {
        start_ids[i] = i * quotient + (i < remainder ? i : remainder);
        num_iters_array[i] = quotient + (i < remainder ? 1 : 0);
    }

    int local_counts_array[number_of_threads];
    int local_counts_array_bond_r[number_of_threads];

    #pragma omp parallel private(i, j, cube_index_temp, link_index, d_temp2) num_threads(number_of_threads)
    {
        int thread_id = omp_get_thread_num();
        int local_count = 0;
        int local_count_bond_r = 0;

        for (i = start_ids[thread_id]; i < start_ids[thread_id] + num_iters_array[thread_id]; i++) {
            for (j = 0; j < 27; j++) {
                if (neighbor_map[center_indices1[i]][j] == -1) {
                    continue;
                }
                cube_index_temp = neighbor_map[center_indices1[i]][j];
                link_index = head[cube_index_temp];
                while (link_index != -1) {
                    d_temp2 = distance2(expanded_coords, center_coords, link_index, i, 3);

                    if (d_temp2 < r2 + tol && i != indices[link_index] && d_temp2 > tol) {
                        local_count += 1;

                        if (d_temp2 < bond_r2 + tol) {
                            local_count_bond_r += 1;
                        }
                    }

                    link_index = atom_indices[link_index];
                }
            }
        }

        local_counts_array[thread_id] = local_count;
        local_counts_array_bond_r[thread_id] = local_count_bond_r;
    }


    // Calculate max_count, also get prefix list indicating where each thread should start placing their values
    // also do this for the within_bond_r array
    max_count = 0;
    int max_count_bond_r = 0;

    int prefix_index_list[number_of_threads];
    int prefix_index_bond_r_list[number_of_threads];

    int running_sum = 0;
    int running_sum_bond_r = 0;

    for (i = 0; i < number_of_threads; i++) {
        prefix_index_list[i] = running_sum;
        prefix_index_bond_r_list[i] = running_sum_bond_r;

        running_sum += local_counts_array[i];
        running_sum_bond_r += local_counts_array_bond_r[i];

        max_count += local_counts_array[i];
        max_count_bond_r += local_counts_array_bond_r[i];
    }

    #ifdef TIMING
        t5 = omp_get_wtime();
        elapsed = t5 - t4;
        printf("Third loop elapsed: %.9f seconds\n", elapsed);
    #endif

    // Allocate memory for output based on pre-calculated max_count
    long *index_1 = (long *)safe_malloc(max_count * sizeof(long));
    long *index_2 = (long *)safe_malloc(max_count * sizeof(long));
    double *offset_final = (double *)safe_malloc(3 * max_count * sizeof(double));
    double *distances = (double *)safe_malloc(max_count * sizeof(double));
    long *within_bond_r_indices = (long *)safe_malloc(max_count_bond_r * sizeof(long));
    long num_within_bond_r_indices = max_count_bond_r;

    #pragma omp parallel private(cube_index_temp, link_index, i, j, d_temp2) num_threads(number_of_threads)
    {
        int thread_id = omp_get_thread_num();
        int local_count = 0;
        int local_count_bond_r = 0;
        int prefix = prefix_index_list[thread_id];
        int global_index;

        for (i = start_ids[thread_id]; i < start_ids[thread_id] + num_iters_array[thread_id]; i++) {
            for (j = 0; j < 27; j++) {
                if (neighbor_map[center_indices1[i]][j] == -1) {
                    continue;
                }

                cube_index_temp = neighbor_map[center_indices1[i]][j];
                link_index = head[cube_index_temp];
                while (link_index != -1) {
                    d_temp2 = distance2(expanded_coords, center_coords, link_index, i, 3);

                    if (d_temp2 < r2 + tol && i != indices[link_index] && d_temp2 > tol) {
                        global_index = prefix + local_count;

                        index_1[global_index] = i;
                        index_2[global_index] = indices[link_index];
                        offset_final[3*global_index] = offsets[3*link_index] - offset_correction[3*indices[link_index]];
                        offset_final[3*global_index + 1] = offsets[3*link_index + 1] - offset_correction[3*indices[link_index] + 1];
                        offset_final[3*global_index + 2] = offsets[3*link_index + 2] - offset_correction[3*indices[link_index] + 2];
                        distances[global_index] = sqrt(d_temp2);

                        if (d_temp2 < bond_r2 + tol) {
                            within_bond_r_indices[prefix_index_bond_r_list[thread_id] + local_count_bond_r] = global_index;
                            local_count_bond_r += 1;
                        }

                        local_count += 1;
                    }

                    link_index = atom_indices[link_index];
                }

            }
        }
    }

    #ifdef TIMING
        t6 = omp_get_wtime();
        elapsed = t6 - t5;
        printf("Fourth loop elapsed: %.9f seconds\n", elapsed);
    #endif


    if (failed_malloc) {
        fprintf(stderr, "A realloc of memory failed!\n");
        exit(EXIT_FAILURE);
    } else {
        failed_malloc = 0;
    }

    *out_index_1 = index_1;
    *out_index_2 = index_2;
    *out_offsets = offset_final;
    *out_distances = distances;
    *out_num_edges = max_count;
    *out_within_bond_r_indices = within_bond_r_indices;
    *out_num_within_bond_r_indices = num_within_bond_r_indices;

    // Free allocated memories
    free(offsets_p);
    free(expanded_coords_p);
    free(indices_p);
    free(head);
    free(atom_indices);

    free(offset_correction);
    free(frac_coords);
    free(all_fcoords);
    free(coords_in_cell);
    free(center_indices1);
    free(center_indices3);

    for (i = 0; i < nb_cubes; i++) {
        free(neighbor_map[i]);
    }
    free(neighbor_map);

    free(all_indices3);
    free(all_indices1);

    return;
}
