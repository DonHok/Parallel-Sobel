#include <getopt.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ppp/ppp.h"
#include "ppp_pnm/ppp_pnm.h"

static void usage(const char *progname) {
    fprintf(stderr,
            "USAGE: %s -i input.pgm [-o output.pgm] [-c coeff] "
            "    [-n] [-v] [-s] [-r] [-N n] [-K kappa] [-D dt] [-E eps] "
            "    [-d] [-h]\n"
            "  coeff  sobel coefficient (default: 0.9)\n"
            "  n      repetitions for VCD (default: 40)\n"
            "  kappa  VCD parameter kappa (default: 30)\n"
            "  dt     VCD parameter dt (default: 0.1)\n"
            "  eps    VCD parameter epsilon (default: 0.005)\n"
            "  -n     use naive sequential implementation\n"
            "  -v     run VCD operator\n"
            "  -s     run Sobel operator (after VCD with -v)\n"
            "  -r     VCD implementation with value reuse and exp approximation\n"
            "  -d     give some debug output\n"
            "  -h     print this help\n",
            progname);
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if (provided < MPI_THREAD_SERIALIZED) {
        fprintf(stderr, "Error: MPI library does not support threads.\n");
        return 1;
    }

    // Default parameter values
    struct TaskInput TI;
    TI.filename = NULL;
    TI.outfilename = NULL;
    TI.doVCD = false;
    TI.doSobel = false;
    TI.improvedVCD = false;
    TI.sobelC = 0.9;
    TI.vcdN = 40;
    TI.vcdEpsilon = 0.005;
    TI.vcdKappa = 30;
    TI.vcdDt = 0.1;
    TI.debugOutput = false;
    bool naive = false;

    int option;
    while ((option = getopt(argc,argv,"i:o:c:N:K:D:E:nvsrdh")) != -1) {
	switch(option) {
	case 'i': TI.filename = strdup(optarg); break;
	case 'o': TI.outfilename = strdup(optarg); break;
	case 'c': TI.sobelC = atof(optarg); break;
	case 'N': TI.vcdN = atoi(optarg); break;
	case 'K': TI.vcdKappa = atof(optarg); break;
	case 'D': TI.vcdDt = atof(optarg); break;
	case 'E': TI.vcdEpsilon = atof(optarg); break;
        case 'n': naive = true; break;
        case 'v': TI.doVCD = true; break;
	case 's': TI.doSobel = true; break;
        case 'r': TI.improvedVCD = true; break;
	case 'd': TI.debugOutput = true; break;
	case 'h':
	    usage(argv[0]);
	    MPI_Finalize();
	    return 0;
	default:
	    usage(argv[0]);
	    MPI_Finalize();
	    return 1;
	}
    }
    if (TI.filename == NULL) {
        usage(argv[0]);
        MPI_Finalize();
	return 1;
    }

    if (naive)
        compute_single(&TI);
    else
        compute_parallel(&TI);

    free(TI.outfilename);
    free(TI.filename);
    
    MPI_Finalize();

    return 0;
}
