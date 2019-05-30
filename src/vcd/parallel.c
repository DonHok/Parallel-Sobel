/*******************************************************************************************************************
 *Task h:                                                                                                          *
 *Setup: Laptop with 6th gen Intel i3 CPU and 16 GB LDDR3 mem O3 optimization. 2 MPI threads, True Marble.         *
 *                                                                                                                 *
 *Result in the form of: Execution time, IPC, CPU usage, D refs, D1 misses, LLd misses                             *
 *Result 1: Improved version of VCD + Sobel                                                                        *
 *  8.356781s, 0.80 IPC, 2.074 CPUs utilized, 28,510,051 (7m rd, 21m wr), 601,529(291k rd+309k wr), 333,903        *
 *                                                                                                                 *
 *Result 2: Normal VCD + sobel                                                                                     *
 *  69.754868s, 1.37 IPC, 3.56 CPUs utilized, 28,506,053  (7m rd + 21m wr), 595,779(287k rd + 308k wr), 333,277    *
 *                                                                                                                 *
 *Answer 1: Improved implementation is more memory bound(usage of additional buffers) as can be seen by the        *
 * low IPC and low CPU utilization(50% is stalled). Another indicator is that the number of reads                  *
 * and writes (and misses) roughly stays the same as in the  Result2, while performing those in a much shorter     *
 * time frame compared to that. Meaning the data throughput should be around 10 times the other implementation.    *
 *                                                                                                                 *
 *Answer 2: This implementation is more compute bound(at least on the tested system).                              *
 * The CPU utilization is high(only 2 physical cores available) and despite having roughly similar cache behavior  *
 * as Result 1 the computation takes nearly 10 times more time. Another indicator:                                 *
 * Approximate the exp call(purely compute bound) and the execution time decreases several times.                  *
 *******************************************************************************************************************
 */

#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

double *left, *right, *top, *bottom;
int row_offset, col_offset, gg_rows, gg_cols, l_columns, l_rows, np;
MPI_Request recv_sides[2];
MPI_Request recv_top[3];
MPI_Request recv_bot[3];
MPI_Request s_req[8];
int sent = 0;
int to_recv_sides = 0;
int to_recv_top = 0;
int to_recv_bot = 0;
int n_col_blocks = 1;
int n_row_blocks = 1;

inline static int calculateOffset(int p_id, int length, int n_blocks) {
    if (n_blocks == 1)
        return 0;
    return (p_id * (length / n_blocks)) + (p_id < length % n_blocks ? p_id : (length % n_blocks));
}

inline static int calculateLength(int p_id, int length, int n_blocks) {
    return (length / n_blocks) + (p_id < length % n_blocks ? 1 : 0);
}

inline static double *dalloc_checked(int size) {
    double *alloced = malloc(sizeof(double) * size);
    if (alloced == NULL) {
        fprintf(stderr, "Could not alloc memory for frame.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return alloced;
}

/* allocate with dummy values at idx -1 and length */
inline static double *alloc_with_dummy(int size) {
    double *allocated = dalloc_checked(size + 2);
    allocated[0] = 0;
    allocated[size + 1] = 0;
    return &allocated[1];
}

inline static void prepareNeighbourBuffers(int self) {
    if (self / n_col_blocks > 0)
        top = alloc_with_dummy(l_columns);
    if (self / n_col_blocks < n_row_blocks - 1)
        bottom = alloc_with_dummy(l_columns);
    if (self % n_col_blocks > 0)
        left = dalloc_checked(l_rows);
    if (self % n_col_blocks < n_col_blocks - 1)
        right = dalloc_checked(l_rows);
}

/* Order the 2d blocks so that they are linear or restore the normal order
 * Result of the reordering is saved in img_in */
inline static void reorderImage(uint8_t **img_in, int np, int reverse, int *sendcounts, int *displs) {
    int reordered = 0;
    uint8_t *tmp = malloc(sizeof(uint8_t) * gg_rows * gg_cols);
    if (tmp == NULL) {
        fprintf(stderr, "Could not allocate memory for reordering image.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < np; i++) {
        // size of 2D block
        int row_block_id = i / n_col_blocks;
        int col_block_id = i % n_col_blocks;
        int p_row_length = calculateLength(row_block_id, gg_rows, n_row_blocks);
        int p_col_length = calculateLength(col_block_id, gg_cols, n_col_blocks);
        // offset within the image
        int p_col_offset = calculateOffset(col_block_id, gg_cols, n_col_blocks);
        int p_row_offset = calculateOffset(row_block_id, gg_rows, n_row_blocks);
        for (int j = 0; j < p_row_length; j++) {
            if (reverse) {
                memcpy(&tmp[(p_row_offset * gg_cols) + (j * gg_cols) + (p_col_offset)], &(*img_in)[reordered],
                       p_col_length);
            } else {
                memcpy(&tmp[reordered], &(*img_in)[(p_row_offset * gg_cols) + (j * gg_cols) + (p_col_offset)],
                       p_col_length);
            }
            reordered += p_col_length;
        }
        if (!reverse) {
            sendcounts[i] = p_row_length * p_col_length;
            displs[i] = reordered - p_row_length * p_col_length;
        }
    }
    free(*img_in);
    *img_in = tmp;
}

/* Calculate the number of blocks in height and width */
inline static void calculateNBlocks(int h, int w, int *n_h, int *n_w, int procs) {
    int min_impurity_image = INT32_MAX;
    int min_impurity_squared = INT32_MAX;

    inline void testDecreaseImpurity(int d1, int d2) {
        // impurity image is the number of pixels that cant be evenly distributed
        // impurity squared is the number of additional rows/cols 1 dimension has
        int remainder_height = h % d1;
        int remainder_width = w % d2;
        int impurity_image = remainder_height * w + remainder_width * w;
        int impurity_squared = d1 < d2 ? d2 - d1 : d1 - d2;
        if (impurity_squared <= min_impurity_squared && impurity_image <= min_impurity_image) {
            min_impurity_image = impurity_image;
            min_impurity_squared = impurity_squared;
            *n_h = d1;
            *n_w = d2;
        }
    }
    for (int dim1 = 1; dim1 <= (procs / 2); dim1++) {
        if (procs % dim1 == 0) {
            int dim2 = procs / dim1;
            // always prefer the one with more splits on the rows if both are equally impure
            testDecreaseImpurity(dim1, dim2);
            testDecreaseImpurity(dim2, dim1);
        }
    }
}

/* perform free on a array starting at a certain index if it is not null */
static inline void freeChecked(double *toFree, int idx) {
    if (toFree != NULL)
        free(&toFree[idx]);
}

/* receive data from neighbours that are above or below the current process */
inline static void recv_from_vertical(double *buff, int self, int up) {
    int source = up ? self - n_col_blocks : self + n_col_blocks;
    // receive from top/bot left
    if (source - 1 >= 0 && (source - 1) / n_col_blocks == (source / n_col_blocks))
        MPI_Irecv(&buff[-1], 1, MPI_DOUBLE, source - 1, 0, MPI_COMM_WORLD,
                  up ? &recv_top[to_recv_top++] : &recv_bot[to_recv_bot++]);
    // receive from top/bot
    MPI_Irecv(buff, l_columns, MPI_DOUBLE, source, 0, MPI_COMM_WORLD,
              up ? &recv_top[to_recv_top++] : &recv_bot[to_recv_bot++]);
    // receive from top/bot right
    if (source + 1 < np && (source + 1) / n_col_blocks == (source / n_col_blocks))
        MPI_Irecv(&buff[l_columns], 1, MPI_DOUBLE, source + 1, 0, MPI_COMM_WORLD,
                  up ? &recv_top[to_recv_top++] : &recv_bot[to_recv_bot++]);
}

/* receive all data from neighbours */
inline static void irecv_neighbours(int self) {
    if (top != NULL)
        recv_from_vertical(top, self, 1);
    if (bottom != NULL)
        recv_from_vertical(bottom, self, 0);
    if (left != NULL)
        MPI_Irecv(left, l_rows, MPI_DOUBLE, self - 1, 0, MPI_COMM_WORLD, &recv_sides[to_recv_sides++]);
    if (right != NULL)
        MPI_Irecv(right, l_rows, MPI_DOUBLE, self + 1, 0, MPI_COMM_WORLD, &recv_sides[to_recv_sides++]);
}

inline static void send(int idx, int length, int proc_id, double *image) {
    MPI_Isend(&image[idx], length, MPI_DOUBLE, proc_id, 0, MPI_COMM_WORLD, &s_req[sent++]);
}

/* send outter parts to neighbours of the a process */
inline static void isend_to_neighbours(double *image, int pid, double *l_buff, double *r_buff) {
    int col_block_id = pid % n_col_blocks;
    int row_block_id = pid / n_col_blocks;
    // send to left/right neighbour
    if (col_block_id > 0)
        send(0, l_rows, pid - 1, l_buff);
    if (col_block_id < n_col_blocks - 1)
        send(0, l_rows, pid + 1, r_buff);

    // send to top left
    if (col_block_id > 0 && row_block_id > 0)
        send(0, 1, pid - 1 - n_col_blocks, image);
    // send to top right
    if (col_block_id < n_col_blocks - 1 && row_block_id > 0)
        send(l_columns - 1, 1, pid + 1 - n_col_blocks, image);
    // send to top
    if (row_block_id > 0)
        send(0, l_columns, pid - n_col_blocks, image);

    // send to bot
    if (row_block_id < n_row_blocks - 1)
        send((l_rows - 1) * l_columns, l_columns, pid + n_col_blocks, image);
    // send to bot left
    if (row_block_id < n_row_blocks - 1 && col_block_id > 0)
        send((l_rows - 1) * l_columns, 1, pid + n_col_blocks - 1, image);
    // send to bot right
    if (row_block_id < n_row_blocks - 1 && col_block_id < n_col_blocks - 1)
        send((l_rows) * l_columns - 1, 1, pid + n_col_blocks + 1, image);
}

/* Access image with bounds/MPI check */
static inline double SC(int c, int r, double *img, int rows, int cols) {
    if (r >= 0 && r < rows && c >= 0 && c < cols) {
        return img[r * cols + c];
        // For both accessing top or bottom at col -1 is defined, see alloc with dummy
    } else if (r < 0) {
        return top != NULL ? top[c] : 0;
    } else if (r >= rows) {
        return bottom != NULL ? bottom[c] : 0;
    } else if (c < 0) {
        return left != NULL ? left[r] : 0;
    } else {
        return right != NULL ? right[r] : 0;
    }
}

inline static void wait_for_bottom_neighbours() {
    MPI_Waitall(to_recv_bot, recv_bot, MPI_STATUSES_IGNORE);
    to_recv_bot = 0;
}

inline static void wait_for_top_neighbours() {
    MPI_Waitall(to_recv_top, recv_top, MPI_STATUSES_IGNORE);
    to_recv_top = 0;
}

inline static void wait_for_horizontal_neighbours() {
    MPI_Waitall(to_recv_sides, recv_sides, MPI_STATUSES_IGNORE);
    to_recv_sides = 0;
}

inline static void send_receive_borders(int self, double *image, double *l_buff, double *r_buff) {
    irecv_neighbours(self);
    isend_to_neighbours(image, self, l_buff, r_buff);
    // Wait until everything has been copied to a buffer
    MPI_Waitall(sent, s_req, MPI_STATUSES_IGNORE);
    sent = 0;
}

/* Compute sobel for a pixel with performing all bounds checks */
inline static double run_sobel_checked(int row, int col, double *img, int rows, int cols, double c_coeff) {
    double sx, sy;
#define EXTRA_ARGS  img, rows, cols
    sx = SC(col - 1, row - 1, EXTRA_ARGS) + 2 * SC(col, row - 1, EXTRA_ARGS) + SC(col + 1, row - 1, EXTRA_ARGS)
         - SC(col - 1, row + 1, EXTRA_ARGS) - 2 * SC(col, row + 1, EXTRA_ARGS) - SC(col + 1, row + 1, EXTRA_ARGS);
    sy = SC(col - 1, row - 1, EXTRA_ARGS) - SC(col + 1, row - 1, EXTRA_ARGS) + 2 * SC(col - 1, row, EXTRA_ARGS)
         - 2 * SC(col + 1, row, EXTRA_ARGS) + SC(col - 1, row + 1, EXTRA_ARGS) - SC(col + 1, row + 1, EXTRA_ARGS);
    return c_coeff * hypot(sx, sy);
#undef EXTRA_ARGS
}

/*
 * Perform the the Sobel operator on a image part
 * The frame around the image part is stored in global vars left, right, top, bot.
 * Top and bottom both have full length + 2 elements(with pointer pointing at the first element)
 */
inline static void sobelP(double *img, double *buf, double c_coeff) {
    int rows = l_rows;
    int cols = l_columns;

    // Access image without any bounds check
    inline double S(int c, int r) {
        return img[r * cols + c];
    }

    // Only block for the most critical parts before the loop
    wait_for_horizontal_neighbours();

    // Split up the loop so only the outer parts of the image have to be accessed with bounds check
#pragma omp parallel for shared(img, buf)
    for (int row = 0; row < rows; row++) {
        double sx, sy;
        // Run first and last row always checked
        if (row == 0 || row >= rows - 1) {
            if (row == 0) {
                wait_for_top_neighbours();
            } else {
                wait_for_bottom_neighbours();
            }
            for (int col = 0; col < cols; col++) {
                buf[row * cols + col] = run_sobel_checked(row, col, img, rows, cols, c_coeff);
            }
        } else {
            // Assume picture always has at least 2 pixels in a column, as edge detection on 1 pixel makes no sense.
            // Run special checked case for first and lats column
            buf[row * cols] = run_sobel_checked(row, 0, img, rows, cols, c_coeff);
            // No bounds check required for this loop
            for (int col = 1; col < cols - 1; col++) {
                sx = S(col - 1, row - 1) + 2 * S(col, row - 1) + S(col + 1, row - 1)
                     - S(col - 1, row + 1) - 2 * S(col, row + 1) - S(col + 1, row + 1);
                sy = S(col - 1, row - 1) - S(col + 1, row - 1) + 2 * S(col - 1, row)
                     - 2 * S(col + 1, row) + S(col - 1, row + 1) - S(col + 1, row + 1);
                buf[row * cols + col] = c_coeff * hypot(sx, sy);
            }
            buf[row * cols + (cols - 1)] = run_sobel_checked(row, cols - 1, img, rows, cols, c_coeff);
        }
    }
}

/*
 * Perform the the vcd operator on a image part. The same frame setting as Sobel. l_buff and r_buff stores the results
 * of the left and right part of the image part.
 * Task a - e
 */
void vcdP(double **img, double **buf, const struct TaskInput *TI,
          double *l_buff, double *r_buff, int self) {
    // local number of rows and columns
    int rows = l_rows;
    int cols = l_columns;
    // offset to locate the image part within the whole image
    int g_row_offset = row_offset;
    int g_col_offset = col_offset;
    int g_rows = gg_rows;
    int g_cols = gg_cols;
    const double kappa = TI->vcdKappa;
    const double kappa_pow_2_times_two = (kappa * kappa) * 2;
    const double kappa_pow_2_times_four = kappa_pow_2_times_two * 2;
    const double kappa_times_two = kappa * 2;
    const double epsilon = TI->vcdEpsilon;
    const double kappa_times_dt = kappa * TI->vcdDt;
    const int N = TI->vcdN;
    double deltaMax = epsilon + 1.0;

    inline double phi(double nu) {
        // can be simplified by precomputing the denominator. division with kappa after all phi are summed up
        return nu * exp(-nu * nu / kappa_pow_2_times_two);
    }

    inline double xi(double nu) {
        // can be simplified by pre computing the denominator. division with 2 times kappa after all phi are summed up.
        return nu * exp(-nu * nu / kappa_pow_2_times_four);
    }

    int iteration = 0;
    while (iteration++ < N && deltaMax > epsilon) {

        // Only wait for the side communication here
        wait_for_horizontal_neighbours();

        deltaMax = 0;
#pragma omp parallel for reduction(max: deltaMax)
        for (int row = 0; row < rows; row++) {
            if (row == 0) {
                wait_for_top_neighbours();
            } else if (row == rows - 1) {
                wait_for_bottom_neighbours();
            }
            for (int col = 0; col < cols; col++) {
#define EXTRA_ARGS  *img, rows, cols
                double delta;
                // do division with kappa only here
                delta = ((phi(SC(col + 1, row, EXTRA_ARGS) - SC(col, row, EXTRA_ARGS))
                          - phi(SC(col, row, EXTRA_ARGS) - SC(col - 1, row, EXTRA_ARGS))
                          + phi(SC(col, row + 1, EXTRA_ARGS) - SC(col, row, EXTRA_ARGS))
                          - phi(SC(col, row, EXTRA_ARGS) - SC(col, row - 1, EXTRA_ARGS))) / kappa)
                        + ((xi(SC(col + 1, row + 1, EXTRA_ARGS) - SC(col, row, EXTRA_ARGS))
                            - xi(SC(col, row, EXTRA_ARGS) - SC(col - 1, row - 1, EXTRA_ARGS))
                            + xi(SC(col - 1, row + 1, EXTRA_ARGS) - SC(col, row, EXTRA_ARGS))
                            - xi(SC(col, row, EXTRA_ARGS) - SC(col + 1, row - 1, EXTRA_ARGS))) / (kappa_times_two));
                (*buf)[row * cols + col] = SC(col, row, EXTRA_ARGS) + kappa_times_dt * delta;
                if (row + g_row_offset > 0 && row + g_row_offset < g_rows - 1
                    && col + g_col_offset > 0 && col + g_col_offset < g_cols - 1) {
                    delta = fabs(delta);
                    if (delta > deltaMax)
                        deltaMax = delta;
                }
#undef EXTRA_ARGS
            }
            if (row >= 0 && row < rows) {
                if (l_buff != NULL)
                    l_buff[row] = (*buf)[row * cols];
                if (r_buff != NULL)
                    r_buff[row] = (*buf)[row * cols + cols - 1];
            }
        }
        MPI_Allreduce(&deltaMax, &deltaMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        // Only send only if required
        if ((iteration < N && deltaMax > epsilon) || TI->doSobel) {
            send_receive_borders(self, *buf, l_buff, r_buff);
        }

        double *tmp = *buf;
        *buf = *img;
        *img = tmp;

        if (TI->debugOutput && self == 0)
            printf("Iteration %2d: max. Delta = %g\n", iteration, deltaMax);
    }
}

/* Taylor series expansion of exp */
inline static double exp_approx(double x) {
    return 1 + x + (x * x) / 2 + (x * x * x) / 6;
}

/* Perform the the optimized vcd operator on a image part. Task f & g. */
void vcdPOptimized(double **img, double **buf, const struct TaskInput *TI,
                   double *l_buff, double *r_buff, int self) {
    // local number of rows and columns
    int rows = l_rows;
    int cols = l_columns;
    // offset to locate the image part within the whole image
    int g_row_offset = row_offset;
    int g_col_offset = col_offset;
    int g_rows = gg_rows;
    int g_cols = gg_cols;
    const double kappa = TI->vcdKappa;
    const double kappa_pow_2_times_two = (kappa * kappa) * 2;
    const double kappa_pow_2_times_four = kappa_pow_2_times_two * 2;
    const double kappa_times_two = kappa * 2;
    const double epsilon = TI->vcdEpsilon;
    const double kappa_times_dt = kappa * TI->vcdDt;
    const int N = TI->vcdN;
    double deltaMax = epsilon + 1.0;

    // Only store precomputed results of the calculation that are on the row above.
    // Pass the other value as variable.
    // Wastes a lot of space but assuming Open MPI is used those costs are decreased due to smaller blocks.
    // Doing this on a machine with a core i3 results in a marginal improvement performance when
    // running only 1 MPI process. However in case all 2 cores are fully used performance increases the performance
    // by 30%.
    double *result_buff = malloc(sizeof(double) * 3 * rows * cols);

    inline double phi(double nu) {
        return nu * exp_approx(-nu * nu / kappa_pow_2_times_two);
    }

    inline double xi(double nu) {
        return nu * exp_approx(-nu * nu / kappa_pow_2_times_four);
    }

    // Access image without any bounds check
    inline double S(int idx) {
        return (*img)[idx];
    }

    // Precompute values(vertical and crosswise) with all required bounds checks
    // only values for parts where the current pixel is subtracted from something are computed
    inline void precomputeChecked(int row, int col) {
#define EXTRA_ARGS  *img, rows, cols
        double current_pixel = S(col + cols * row);
        int buff_idx = (row * cols + col) * 3;
        result_buff[buff_idx] = phi(SC(col, row + 1, EXTRA_ARGS) - current_pixel);
        result_buff[buff_idx + 1] = xi(SC(col + 1, row + 1, EXTRA_ARGS) - current_pixel);
        result_buff[buff_idx + 2] = xi(SC(col - 1, row + 1, EXTRA_ARGS) - current_pixel);
#undef EXTRA_ARGS
    }

    /* Compute delta with all required bounds checks reusing precomputed results */
    inline double runChecked(int row, int col, double *deltaMax, double previous) {
        double current_pixel = S(col + cols * row);
        int own_results = (row * cols + col) * 3;
        double tmp = phi(SC(col + 1, row, *img, rows, cols) - current_pixel);
        double phis = tmp - previous + result_buff[own_results];
        if (row > 0) {
            phis = (phis - result_buff[((row - 1) * cols + col) * 3]);
        } else {
            phis = (phis - phi(current_pixel - SC(col, row - 1, *img, rows, cols)));
        }
        double xis = result_buff[own_results + 1] + result_buff[own_results + 2];
        if (row > 0 && col > 0) {
            xis = xis - result_buff[((row - 1) * cols + col - 1) * 3 + 1];
        } else {
            xis = xis - xi(current_pixel - SC(col - 1, row - 1, *img, rows, cols));
        }
        if (row > 0 && col < cols - 1) {
            xis = xis - result_buff[((row - 1) * cols + col + 1) * 3 + 2];
        } else {
            xis = xis - xi(current_pixel - SC(col + 1, row - 1, *img, rows, cols));
        }
        double delta = ((phis / kappa) + (xis / kappa_times_two));
        (*buf)[col + cols * row] = current_pixel + kappa_times_dt * delta;
        if (row + g_row_offset > 0 && row + g_row_offset < g_rows - 1
            && col + g_col_offset > 0 && col + g_col_offset < g_cols - 1) {
            delta = fabs(delta);
            if (delta > *deltaMax)
                *deltaMax = delta;
        }
        return tmp;
    }

    // Making a distinction between unchecked and checked parts and computing which one has to be used
    // in the first loop reduces the number of branches by 30% and the number of instructions by 10%.
    // This generally results in approx 7% increase in performance on a 2 core system full open mp usage
    // and 1 mpi process at close to no additional memory cost.
    // Deleting additional unrequired checks due to this and accessing indexes better further reduces the number of
    // instructions by 7-9%.
    int iteration = 0;
    while (iteration++ < N && deltaMax > epsilon) {

        wait_for_horizontal_neighbours();
        deltaMax = 0;

        // Precompute intermediate results
#pragma omp parallel for
        for (int row = 0; row < rows; row++) {
            if (row == 0 || row >= rows - 1) {
                if (row == 0) {
                    wait_for_top_neighbours();
                } else {
                    wait_for_bottom_neighbours();
                }
                for (int col = 0; col < cols; col++) {
                    precomputeChecked(row, col);
                }
            } else {
                for (int col = 1; col < cols - 1; col++) {
                    double current_pixel = S(row * cols + col);
                    int buff_idx = (row * cols + col) * 3;
                    result_buff[buff_idx] = phi(S((row + 1) * cols + col) - current_pixel);
                    result_buff[buff_idx + 1] = xi(S((row + 1) * cols + col + 1) - current_pixel);
                    result_buff[buff_idx + 2] = xi(S((row + 1) * cols + col - 1) - current_pixel);
                }
                // Run rest checked
                precomputeChecked(row, 0);
                precomputeChecked(row, cols - 1);
            }
        }

        // Run actual VCD
#pragma omp parallel for reduction(max: deltaMax)
        for (int row = 0; row < rows; row++) {
            // pass previous values on the same column along the iterations
            double previous = phi(S(row * cols) - SC(-1, row, *img, rows, cols));

            if (row == 0 || row >= rows - 1) {
                for (int col = 0; col < cols; col++) {
                    previous = runChecked(row, col, &deltaMax, previous);
                }
                if (l_buff != NULL)
                    l_buff[row] = (*buf)[row * cols];
                
                if (r_buff != NULL)
                    r_buff[row] = (*buf)[row * cols + cols - 1];
            } else {
                previous = runChecked(row, 0, &deltaMax, previous);
                if (l_buff != NULL)
                    l_buff[row] = (*buf)[row * cols];
                // Run this part unchecked
                for (int col = 1; col < cols - 1; col++) {
                    double current_pixel = S(col + row * cols);
                    int own_results = (row * cols + col) * 3;
                    double tmp = phi(S(col + 1 + row * cols) - current_pixel);
                    double delta = ((tmp - previous + result_buff[own_results]
                                     - result_buff[((row - 1) * cols + col) * 3]) / kappa)
                                   + ((result_buff[own_results + 1]
                                       - result_buff[((row - 1) * cols + col - 1) * 3 + 1]
                                       + result_buff[own_results + 2]
                                       - result_buff[((row - 1) * cols + col + 1) * 3 + 2]) / kappa_times_two);
                    (*buf)[col + cols * row] = current_pixel + kappa_times_dt * delta;
                    // This condition can never be false for these iterations as this loop is always
                    // only performed on the inner parts of the of an image / image part
                    // Reduces the number of instructions by another approx 1.5% at no cost.
                    /*      if (row + g_row_offset > 0 && row + g_row_offset < g_rows - 1
                              && col + g_col_offset > 0 && col + g_col_offset < g_cols - 1) { */
                    delta = fabs(delta);
                    if (delta > deltaMax)
                        deltaMax = delta;
                    previous = tmp;
                }
                runChecked(row, cols - 1, &deltaMax, previous);
                if (r_buff != NULL)
                    r_buff[row] = (*buf)[row * cols + cols - 1];
            }
        }

        MPI_Allreduce(&deltaMax, &deltaMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (TI->debugOutput && self == 0)
            printf("Iteration %2d: max. Delta = %g\n", iteration, deltaMax);

        if ((iteration < N && deltaMax > epsilon) || TI->doSobel) {
            send_receive_borders(self, *buf, l_buff, r_buff);
        }
        double *tmp = *buf;
        *buf = *img;
        *img = tmp;
    }
    free(result_buff);
}

void compute_parallel(const struct TaskInput *TI) {
    enum pnm_kind kind;
    int grow_cols[2], maxcolor, self;
    uint8_t *image;

    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &self);

    if (self == 0) {
        printf("Number of MPI processes: %d\n", np);
#pragma omp parallel
        {
#pragma omp single
            printf("Number of OMP threads in each MPI process: %d\n",
                   omp_get_num_threads());
        }
    }

    int *sendcnt = malloc(sizeof(int) * np);
    int *displs = malloc(sizeof(int) * np);
    if ((sendcnt == NULL) || (displs == NULL)) {
        fprintf(stderr, "Could not initialize sendcts/displs.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (self == 0) {
        image = ppp_pnm_read(TI->filename, &kind, &grow_cols[0], &grow_cols[1], &maxcolor);
        if (TI->debugOutput)
            printf("Dimensions: rows: %d cols: %d\n", grow_cols[0], grow_cols[1]);
        if ((image == NULL) | (kind != PNM_KIND_PGM)) {
            fprintf(stderr, "Could not load image from file '%s'.\n", TI->filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Bcast(&grow_cols, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&maxcolor, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // global number of rows and cols
    gg_rows = grow_cols[0];
    gg_cols = grow_cols[1];

    // calculate how the image is split up
    calculateNBlocks(gg_rows, gg_cols, &n_row_blocks, &n_col_blocks, np);
    if (self == 0) {
        // prepare and send image when it is in uint8_t format as it saves a lot of bandwidth/space
        reorderImage(&image, np, 0, sendcnt, displs);
    }

    if (TI->debugOutput && self == 0) {
        printf("Number of blocks in a row: %d. Number of blocks in a column: %d.\n", n_col_blocks, n_row_blocks);
    }

    // number of rows and cols of this process
    l_rows = calculateLength(self / n_col_blocks, gg_rows, n_row_blocks);
    l_columns = calculateLength(self % n_col_blocks, gg_cols, n_col_blocks);

    if (self != 0) {
        image = malloc(sizeof(uint8_t) * l_rows * l_columns);
        if (image == NULL) {
            fprintf(stderr, "Could not allocate memory for image.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // offset if the partial image within the complete image
    col_offset = calculateOffset(self % n_col_blocks, gg_cols, n_col_blocks);
    row_offset = calculateOffset(self / n_row_blocks, gg_rows, n_row_blocks);

    // Prepare buffers for data received from neighbours
    right = left = bottom = top = NULL;
    prepareNeighbourBuffers(self);
    MPI_Scatterv(image, sendcnt, displs, MPI_UINT8_T, (self == 0 ? MPI_IN_PLACE : image), l_rows * l_columns,
                 MPI_UINT8_T, 0, MPI_COMM_WORLD);

    double *image_partD = (double *) malloc(sizeof(double) * l_rows * l_columns);
    double *temp_partD = (double *) malloc(sizeof(double) * l_rows * l_columns);
    if (image_partD == NULL || temp_partD == NULL) {
        fprintf(stderr, "Could not allocate memory for the image\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

#pragma omp parallel for
    for (int i = 0; i < l_rows * l_columns; ++i) {
        image_partD[i] = (double) image[i] / maxcolor;
    }

    double *l_buff = NULL, *r_buff = NULL;
    if (left != NULL) {
        l_buff = malloc(sizeof(double) * l_rows);
    }
    if (right != NULL) {
        r_buff = malloc(sizeof(double) * l_rows);
    }

    // Initialize buffers for data that will be sent to neighbours
#pragma omp parallel for
    for (int i = 0; i < l_rows; ++i) {
        if (right != NULL)
            r_buff[i] = image_partD[i * l_columns + (l_columns - 1)];
        if (left != NULL) {
            l_buff[i] = image_partD[i * l_columns];
        }
    }

    irecv_neighbours(self);
    isend_to_neighbours(image_partD, self, l_buff, r_buff);

    double time_loaded = seconds();

    if (TI->doVCD) {
        if (TI->improvedVCD) {
            vcdPOptimized(&image_partD, &temp_partD, TI, l_buff, r_buff, self);
        } else {
            vcdP(&image_partD, &temp_partD, TI, l_buff, r_buff, self);
        }
    }

    if (TI->doSobel) {
        sobelP(image_partD, temp_partD, TI->sobelC);
        double *tmp = image_partD;
        image_partD = temp_partD;
        temp_partD = tmp;
    }

    double time_computed = seconds();

    freeChecked(right, 0);
    freeChecked(left, 0);
    freeChecked(top, -1);
    freeChecked(bottom, -1);

#pragma omp parallel for
    for (int i = 0; i < l_rows * l_columns; ++i) {
        int v = lrint(image_partD[i] * maxcolor);
        image[i] = (v < 0 ? 0 : (v > maxcolor ? maxcolor : v));
    }

    free(image_partD);
    free(temp_partD);

    MPI_Gatherv((self == 0 ? MPI_IN_PLACE : image), l_rows * l_columns, MPI_UINT8_T, image,
                sendcnt, displs, MPI_UINT8_T, 0, MPI_COMM_WORLD);

    if (TI->outfilename != NULL && self == 0) {
        reorderImage(&image, np, 1, sendcnt, displs);
        if (ppp_pnm_write(TI->outfilename, kind, gg_rows, gg_cols, maxcolor,
                          image) == -1) {
            fprintf(stderr, "Could not write output to '%s'.\n", TI->outfilename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    free(image);
    printf("Computation time: %.6f\n", time_computed - time_loaded);
}