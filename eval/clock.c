/* clock.c
 * Retrofitted to use thread-specific timers
 * and to get clock information from /proc/cpuinfo
 * (C) R. E. Bryant, 2010
 *
 * Retrofitted to measure absolute time in seconds and then convert to cycles
 * Old time stamp could removed, since time stamp counter no longer tracks clock cycles
 * (C) R. E. Bryant, 2016
 *
 * Simplified and updates.
 * (C) R. E. Bryant, 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "clock.h"

int gverbose = 1;

/* Get megahertz from /etc/proc */
#define MAXBUF 512
double cpu_mhz_full(bool verbose, int *idp) {
    /* Assume that the core running this program has the highest clock rate */
    static char buf[MAXBUF];
    FILE *fp = fopen("/proc/cpuinfo", "r");
    double cpu_mhz = 0.0;
    int current_proc = -1;
    int fastest_proc = -1;

    if (!fp) {
	if (verbose)
	    fprintf(stderr, "Can't open /proc/cpuinfo to get clock information\n");
	cpu_mhz = 1000.0;
	if (idp)
	    *idp = fastest_proc;
	return cpu_mhz;
    }
    while (fgets(buf, MAXBUF, fp)) {
	if (strstr(buf, "cpu MHz")) {
	    double nmhz = 0.0;
	    if (sscanf(buf, "cpu MHz\t: %lf", &nmhz) == 1) {
		current_proc++;
		//		printf("\nGot %.2f MHz on CPU %d\n", nmhz, current_proc);
		if (nmhz > cpu_mhz) {
		    cpu_mhz = nmhz;
		    fastest_proc = current_proc;
		}
		
	    }
	}
    }
    fclose(fp);
    if (cpu_mhz == 0.0) {
	if (verbose)
	    fprintf(stderr, "Can't open /proc/cpuinfo to get clock information\n");
	cpu_mhz = 1000.0;
	if (idp)
	    *idp = fastest_proc;
	return cpu_mhz;
    }
    if (verbose) {
	printf("Processor Clock Rate ~= %.4f GHz (extracted from file) running on processor %d\n", cpu_mhz * 0.001, fastest_proc);
    }
    if (idp)
	*idp = fastest_proc;
    return cpu_mhz;
}

double cpu_mhz() {
    return cpu_mhz_full(false, NULL);
}

/* Use nanosecond timer */
struct timespec last_time;
struct timespec new_time;

/* Use thread clock */
#define CLKT CLOCK_THREAD_CPUTIME_ID


void start_timer()
{
    int rval;
    rval = clock_gettime(CLKT, &last_time);
    if (rval != 0) {
	fprintf(stderr, "Couldn't get time\n");
	exit(1);
    }
}

double get_timer()
{
    int rval;
    double delta_secs = 0.0;
    rval = clock_gettime(CLKT, &new_time);
    if (rval != 0) {
	fprintf(stderr, "Couldn't get time\n");
	return 1e20;
    }
    delta_secs = 1.0 * (new_time.tv_sec - last_time.tv_sec) + 1e-9 * (new_time.tv_nsec - last_time.tv_nsec);
    return delta_secs;
}


