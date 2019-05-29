#include <math.h>
#include <stdio.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

/*
 * Convert a gray value in the range 0..maxcolor to a double value in [0,1].
 */
inline static double grayvalueToDouble(uint8_t v, int maxcolor) {
    return (double)v / maxcolor;
}

/*
 * Convert a double value back to a gray value in the range 0..maxcolor.
 */
inline static int grayvalueFromDouble(double d, int maxcolor) {
    int v = lrint(d * maxcolor);
    return (v < 0 ? 0 : (v > maxcolor ? maxcolor : v));
}

/*
 * Swap the values pointed to by 'a' and 'b',
 * i.e., swap the two double pointers that 'a' and 'b'
 * point to.
 */
static void swap(double **a, double **b) {
    double *temp = *a;
    *a = *b;
    *b = temp;
}

/*
 * VCD operator (see project description for details).
 * The parameters 'image' and 'temp' are pointers to the variables
 * that store a pointer to the image and a temporary buffer because
 * we swap the pointers after each iteration so the output is
 * again in the variable the pointer 'image' points to.
 */
void vcd(double **image, double **temp, int rows, const int columns,
         const struct TaskInput *TI) {
    const double kappa   = TI->vcdKappa;
    const double epsilon = TI->vcdEpsilon;
    const double dt      = TI->vcdDt;
    const int N = TI->vcdN;

    /* Local functions phi and xi from the specification;
     * see sobel() below for a comment on local functions in C.
     */
    inline double phi(double nu) {
        const double chi = nu / kappa;
        return chi * exp(-chi*chi/2.0);
    }

    inline double xi(double nu) {
        const double psi = nu / (kappa * sqrt(2.0));
        return 1/sqrt(2.0) * psi * exp(-psi*psi/2.0);
    }

    /* helper function to access the image */
    inline double S(int c, int r) {
        return r >= 0 && r < rows && c >= 0 && c < columns ? (*image)[r*columns+c] : 0;
    }

    int iteration = 0;
    double deltaMax = epsilon + 1.0;
    while (iteration++ < N && deltaMax > epsilon) {
        deltaMax = 0;
#pragma omp parallel for reduction(max: deltaMax)
        for (int y=0; y<rows; y++) {
            for (int x=0; x<columns; x++) {
                double delta;
                delta = phi(S(x+1,y)-S(x,y)) - phi(S(x,y)-S(x-1,y))
                      + phi(S(x,y+1)-S(x,y)) - phi(S(x,y)-S(x,y-1))
                      + xi(S(x+1,y+1)-S(x,y)) - xi(S(x,y)-S(x-1,y-1))
                      + xi(S(x-1,y+1)-S(x,y)) - xi(S(x,y)-S(x+1,y-1));
                (*temp)[y*columns+x] = S(x,y) + kappa*dt*delta;
                if (y > 0 && y < rows-1 && x > 0 && x < columns-1) {
                    delta = fabs(delta);
                    if (delta > deltaMax)
                        deltaMax = delta;
                }
            }
        }

        swap(image, temp);
        if (TI->debugOutput)
            printf("Iteration %2d: max. Delta = %g\n", iteration, deltaMax);
    }
}

/*
 * Sobel Operator
 *
 * The Sobel operator detects edges in a grayscale image by
 * computing a new pixel value t(x,y) for each pixel
 * from the old pixel values s(x,y) by the following formula:
 *
 *   s_x(x,y) =   s(x-1,y-1) + 2*s(x,y-1) + s(x+1,y-1)
 *              - s(x-1,y+1) - 2*s(x,y+1) - s(x+1,y+1)
 *
 *   s_y(x,y) =   s(x-1,y-1) + 2*s(x-1,y) + s(x-1,y+1)
 *              - s(x+1,y-1) - 2*s(x+1,y) - s(x+1,y+1)
 *
 *   t(x,y) = c * sqrt( s_x(x,y)^2 + s_y(x,y)^2 )
 *
 * Values of pixels outside the input image (s(-1,-1) etc.) are
 * considered to be 0.
 *
 * After the computation, 'input' and 'temp' are swapped, so the
 * result image is again in 'input'.
 */
void sobel(double **input, double **temp, int rows, const int columns, double sobelC) {
    /*
     * GCC allows local functions. Without local functions,
     * we would have to use a macro, e.g.,
     *   #define S(c,r) ((r) >= 0 && (r) < rows && (c) >= 0 && (c) < columns ? image[r*columns+c] : 0)
     * to access the values conveniently with bounds checking.
     */
    inline double S(int c, int r) {
        return r >= 0 && r < rows && c >= 0 && c < columns ? (*input)[r*columns+c] : 0;
    }

    for (int y=0; y<rows; ++y) {
	for (int x=0; x<columns; ++x) {
            double sx, sy;
	    sx = S(x-1,y-1) + 2*S(x,y-1) + S(x+1,y-1)
		-S(x-1,y+1) - 2*S(x,y+1) - S(x+1,y+1);
	    sy = S(x-1,y-1) + 2*S(x-1,y) + S(x-1,y+1)
	        -S(x+1,y-1) - 2*S(x+1,y) - S(x+1,y+1);
	    (*temp)[y*columns+x] = sobelC * hypot(sx,sy);
	}
    }

    swap(input, temp);
}


void compute_single(const struct TaskInput *TI) {
    enum pnm_kind kind;
    int rows, columns, maxcolor;
    uint8_t *image;
    
    image = ppp_pnm_read(TI->filename, &kind, &rows, &columns, &maxcolor);

    if (image == NULL) {
	fprintf(stderr, "Could not load image from file '%s'.\n",
		TI->filename);
	exit(1);
    } else if (kind != PNM_KIND_PGM) {
        fprintf(stderr, "Image is not a \"portable graymap.\"\n");
	exit(1);
    }

    // vcd() and sobel() both perform the computation in double values
    // (in the range [0,1]). Both functions read from one array and
    // write the new values to another array.
    double *imageD = (double *)malloc(sizeof(double) * rows*columns);
    double *tempD  = (double *)malloc(sizeof(double) * rows*columns);
    if (imageD == NULL || tempD == NULL) {
        fprintf(stderr, "Could not allocate memory for the image\n");
        exit(1);
    }

    // Copy the original image to imageD and convert it to double values.
    for (int i=0; i<rows*columns; ++i)
        imageD[i] = grayvalueToDouble(image[i], maxcolor);
    
    double time_loaded = seconds();

    // vcd() and sobel() swap the pointers imageD and tempD after each
    // computation step; therefore, we must pass addresses to these
    // variables. The result of the computation is always in imageD.
    
    if (TI->doVCD)
        vcd(&imageD, &tempD, rows, columns, TI);

    if (TI->doSobel)
        sobel(&imageD, &tempD, rows, columns, TI->sobelC);

    double time_computed = seconds();
    
    for (int i=0; i<rows*columns; ++i)
        image[i] = grayvalueFromDouble(imageD[i], maxcolor);

    free(imageD);
    free(tempD);

    if (TI->outfilename != NULL) {
        if (ppp_pnm_write(TI->outfilename, kind, rows, columns, maxcolor,
                          image) == -1) {
            fprintf(stderr, "Could not write output to '%s'.\n", TI->outfilename);
        }
    }

    printf("Computation time: %.6f\n", time_computed-time_loaded);
}
