#include <stdbool.h>
#include <stdlib.h>
#include <sys/time.h>

struct TaskInput {
    // name of the input file
    char *filename;
    
    // name of the output file
    char *outfilename;

    // whether to perform VCD
    bool doVCD;
    
    // whether to perform Sobel (after VCD if doVCD==true)
    bool doSobel;

    // use improved VCD implementation (with value reuse and
    // an approximation for the exp() function)
    bool improvedVCD;

    // parameters for VCD
    int vcdN;
    double vcdEpsilon;
    double vcdKappa;
    double vcdDt;

    // parameters for Sobel
    double sobelC;

    // print some debug outputs during computation
    bool debugOutput;
};

void compute_single(const struct TaskInput *TI);
void compute_parallel(const struct TaskInput *TI);

// Returns the number of seconds since 1970-01-01T00:00:00.
inline static double seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + ((double)tv.tv_usec) / 1000000;
}
