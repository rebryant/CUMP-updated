/* Routines for timing functions */

#include <stdbool.h>

/* Timer: measures in seconds */

/* Start the timer */
void start_timer();

/* Get # seconds since timer started.  Returns 1e20 if detect timing anomaly */
double get_timer();

/* Determine clock rate of processor (using a default sleeptime) */
double cpu_mhz_full(bool verbose, int *cpu_id);

double cpu_mhz();
