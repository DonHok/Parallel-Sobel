#include <stdio.h>
#include <stdlib.h>
#include "ppp_pnm/ppp_pnm.h"

/*
 * Compare two PGM images and report differing pixels.
 *
 */
int main(int argc, char *argv[]) {
    uint8_t *a, *b;
    int rows1, rows2, columns1, columns2, maxcolor1, maxcolor2;
    enum pnm_kind kind1, kind2;
    int x, y;
    int diffs = 0, maxdiff = 0;

    if (argc != 3) {
	fprintf(stderr, "USAGE: %s file1 file2\n", argv[0]);
	return 1;
    }

    a = ppp_pnm_read(argv[1], &kind1, &rows1, &columns1, &maxcolor1);
    if (a == NULL) {
	fprintf(stderr, "Could not open '%s'\n", argv[1]);
	return 1;
    }

    b = ppp_pnm_read(argv[2], &kind2, &rows2, &columns2, &maxcolor2);
    if (b == NULL) {
	fprintf(stderr, "Could not open '%s'\n", argv[2]);
	return 1;
    }

    if (kind1 == kind2 && rows1 == rows2 && columns1 == columns2) {
	if (kind1 == PNM_KIND_PGM) {
	    for (y=0; y<rows1; ++y) {
		for (x=0; x<columns1; ++x) {
	            int absdiff = abs((int)a[y*columns1+x] - (int)b[y*columns1+x]);
		    if (absdiff != 0) {
			printf("At (%d,%d): %d vs %d\n", x, y,
			       a[y*columns1+x], b[y*columns1+x]);
                        ++diffs;
                        if (absdiff > maxdiff)
                           maxdiff = absdiff;
                    }
		}
            }
            printf("Number of differing pixels: %d\n", diffs);
            printf("Maximal difference: %d\n", maxdiff);
	} else {
	    fprintf(stderr, "file are not PGM\n");
	    return 1;
	}
    } else {
	fprintf(stderr, "format mismatch\n");
	return 1;
    }

    return 0;
}
