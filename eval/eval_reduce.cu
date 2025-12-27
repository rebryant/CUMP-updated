#include <stdio.h>
#include <gmp.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "cump/cump.cuh"
#include "clock.h"
#include "mpf_extra.h"

int trials = 5;


bool verbose = false;

double min_value = 0.0;
double max_value = 1.0;

int plist[] = {64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 0};

double gmp_reduce (mpf_t dest, bool product, int  n, int reps, mpf_t  X []);
double cump_reduce (mpf_t dest, bool product, int  n, int reps, int threads, mpf_t  X []);
double double_reduce (double *dval, bool product, int  n, int reps, int threads, mpf_t  X []);

void usage(char *name) {
    printf("Usage: %s [-h] [-v] [-n ELE] [-r REP] [-O p|s|b] [-t THD] [-p PREC] [-s SEED] [-m MIN] [-M MAX]\n", name);
    printf("   -h      Print this message\n");
    printf("   -q      Print CSV data\n");
    printf("   -v      Verbose mode\n");
    printf("   -n ELE  Number of array elements\n");
    printf("   -r REP  Number of times to add/multiply each element\n");
    printf("   -O OPS  Specify which operations to run: Product (p), Sum (s), or Both (b)\n");
    printf("   -t THD  Number of threads\n");
    printf("   -p PREC Precision\n");
    printf("   -s SEED Seed\n");    
    printf("   -m MIN  Minimum value\n");
    printf("   -m MAX  Maximum value\n");
}


int  main (int argc, char *argv[])
{
    int  seed = 341;    /* random seed*/
    int N = 4096;  /* Total number of elements */
    int threads = 256;
    int reps = 100;
    bool do_product = true;
    bool do_sum = true;
    bool csv_only = false;

    int c;
    while ((c = getopt(argc, argv, "hvqn:r:O:t:p:s:m:M:")) != -1) {
	switch(c) {
	case 'h':
	    usage(argv[0]);
	    return 0;
	    break;
	case 'v':
	    verbose = true;
	    break;
	case 'q':
	    csv_only = true;
	    break;
	case 'n':
	    N = atoi(optarg);
	    break;
	case 'r':
	    reps = atoi(optarg);
	    break;
	case 'O':
	    if (optarg[0] == 'p')
		do_sum = false;
	    else if (optarg[0] == 's')
		do_product = false;
	    break;
	case 't':
	    threads = atoi(optarg);
	    break;
	case 'p':
	    /* Force single precision */
	    plist[0] = atoi(optarg);
	    plist[1] = 0;
	    break;
	case 's':
	    seed = atoi(optarg);
	    break;
	case 'm':
	    min_value = atof(optarg);
	    break;
	case 'M':
	    max_value = atof(optarg);
	    break;
	}
    }

    if (min_value > max_value) {
	fprintf(stderr, "Can't have min %.5f > max %.5f\n", min_value, max_value);
	exit(1);
    }

    /* Use maximum precision as reference */
    int max_prec = plist[0];
    for (int pi = 1; plist[pi] != 0; pi++) {
	if (plist[pi] > max_prec)
	    max_prec = plist[pi];
    }
    int test_prec = max_prec + 64;
    mpf_settings_init(test_prec, seed);

    mpf_t *X = uniform_array_mpf(N, min_value, max_value, 0.0, seed);


    if (verbose) {
	printf ("Calculation (vector length = %d, reps = %d, max precision = %d):\n", N, reps, max_prec);
	gmp_printf ("\t\t  |%Ff\t|\n", X [0]);
	gmp_printf ("\t\t  |%Ff\t|\n", X [1]);
	gmp_printf ("\t\t  |    :\t|\n");
	gmp_printf ("\t\t  |    :\t|\n");
	gmp_printf ("\t\t  |%Ff\t|\n\n", X [N-1]);
    }

    if (csv_only)
	printf("Data,Operation,dp,ps\n");
    bool run_product[2];
    mpf_t  cdest[2]; 
    const char *op_name[2];
    int phase_count = 0;
    if (do_sum) {
	op_name[phase_count] = "sum";
	run_product[phase_count++] = false;
    }
    if (do_product) {
	op_name[phase_count] = "product";
	run_product[phase_count++] = true;
    }

    for (int phase = 0; phase < phase_count; phase++) {
	bool product = run_product[phase];
	mpf_init(cdest[phase]);
	/* run reduce */
	double dval;
	if (!csv_only)
	    printf("Computing %s with gmp (precision = %d)\n", op_name[phase], test_prec);
	double gt = gmp_reduce (cdest[phase], product, N, reps, X);
	if (!csv_only) {
	    printf("Computing %s with double\n", op_name[phase]);
	    double dt = double_reduce (&dval, product, N, reps, threads, X);
	    gmp_printf("\t%s\tcpu    = %.20Fg, precision = %d ps_per_op = %.5g\n", op_name[phase], cdest, test_prec, (gt * 1e12)/(N*reps));
	    printf(    "\t%s\tdouble = %.5g, ps_per_op = %.5g\n", op_name[phase], dval, (dt * 1e12)/(N*reps));
	}
    }

    for (int pi = 0; plist[pi] != 0; pi++) {
	int prec = plist[pi];
	mpf_set_default_prec(prec);
	cumpf_set_default_prec(prec);
	for (int phase = 0; phase < phase_count; phase++) {
	    mpf_t gdest;
	    mpf_init(gdest);
	    bool product = run_product[phase];
	    /* run reduce */
	    if (verbose)
		printf("Computing %s with cump.  Precision = %d\n", op_name[phase], prec);
	    double ct = cump_reduce (gdest, product, N, reps, threads, X);
	    double dp = digit_precision(gdest, cdest[phase]);
	    if (!csv_only)
		gmp_printf("\t%s\tgpu    = %.20Fg precision = %d Digit Precision = %.1f ps_per_op = %.5g\n", 
			   op_name[phase], gdest, prec, dp, (ct * 1e12)/(N*reps));
	    if (csv_only) {
		printf("cump%d,%s,%.1f,%.5g\n", prec, op_name[phase], dp, (ct * 1e12)/(N*reps));
	    }
	    mpf_clear(gdest);
	}
    }

    for (int phase = 0; phase < phase_count; phase++)
	mpf_clear(cdest[phase]);


    /* finalize */
    for (int i = 0;  i < N;  ++i)
	{
	    mpf_clear (X [i]);
	}
    return  0;
}


double  gmp_reduce (mpf_t dest, bool product, int  n, int reps, mpf_t  X [])
{
  int  i, r;
  start_timer();
  if (product) {
      mpf_set_d(dest, 1.0);
      for (i = 0;  i < n;  ++i)
	  for (r = 0; r < reps; r++)
	      mpf_mul (dest, dest, X [i]);
  } else {
      mpf_set_d(dest, 0.0);
      for (i = 0;  i < n;  ++i)
	  for (r = 0; r < reps; r++)
	      mpf_add (dest, dest, X [i]);
  }
  return get_timer();
}


using cump::mpf_array_t;

template <bool doMul> __global__
void  cump_reduce_kernel (int  n, int reps, mpf_array_t  X_in, mpf_array_t  X_out, mpf_array_t X_tmp)
{
  using namespace cump;
  cg::thread_block cta = cg::this_thread_block();
  
  int  tid = threadIdx.x;
  int  i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
      mpf_set(X_tmp[i], X_in[i]);
      for (int r = 1; r < reps; r++) {
	  if (doMul)
	      mpf_mul(X_tmp[i], X_tmp[i], X_in[i]);
	  else
	      mpf_add(X_tmp[i], X_tmp[i], X_in[i]);
      }
  }
  cg::sync(cta);

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && i+s < n) {
	  if (doMul)
	      mpf_mul(X_tmp[i], X_tmp[i], X_tmp[i+s]);
	  else
	      mpf_add(X_tmp[i], X_tmp[i], X_tmp[i+s]);
      }
      cg::sync(cta);
  }

  if (tid == 0 && i < n)
      mpf_set(X_out[blockIdx.x], X_tmp[i]);
}

template <bool doMul> __global__
void double_reduce_kernel (int  n, int reps, double  X_in[], double  X_out[], double X_tmp[])
{
  using namespace cump;
  cg::thread_block cta = cg::this_thread_block();
  
  int  tid = threadIdx.x;
  int  i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
      X_tmp[i] = X_in[i];
      for (int r = 1; r < reps; r++) {
	  if (doMul)
	      X_tmp[i] *= X_in[i];
	  else
	      X_tmp[i] += X_in[i];
      }
  }
  cg::sync(cta);


  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s && i+s < n) {
	  if (doMul)
	      X_tmp[i] *= X_tmp[i+s];
	  else
	      X_tmp[i] += X_tmp[i+s];
      }
      cg::sync(cta);
  }

  if (tid == 0 && i < n)
      X_out[blockIdx.x] = X_tmp[i];
}


unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

double  cump_reduce (mpf_t dest, bool product, int  n, int reps, int maxThreads, mpf_t  X[])
{
  cumpf_array_t X_both[2];
  cumpf_array_init(X_both[0], n);
  cumpf_array_t X_tmp;
  cumpf_array_init(X_tmp, n);
  mpf_set_d(dest, product ? 1.0 : 0.0);
  
  int first_threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
  int first_blocks = (n + first_threads-1) / first_threads;
  cumpf_array_init(X_both[1], first_blocks);

  mpf_t R[1];
  mpf_init(R[0]);

  double best_time = 1e9;
  for (int t = 0; t < trials; t++) {
      cumpf_array_set_mpf(X_both[0], X, n);
      cudaDeviceSynchronize();
      start_timer();
      int size = n;
      bool first_time = true;
      int i_source = 0;
      while (first_time || size > 1) {
	  int kreps = first_time ? reps : 1;
	  first_time = false;
	  int threads = (size < maxThreads) ? nextPow2(size) : maxThreads;
	  int blocks = (size + threads-1)/threads;
	  dim3 dimGrid(blocks, 1, 1);
	  dim3 dimBlock(threads, 1, 1);
	  if (verbose) {
	      printf("Running CUMP kernel.  Size = %d, Reps = %d\n", size, kreps);
	  }
	  if (product)
	      cump_reduce_kernel<true><<<dimGrid, dimBlock>>>(size, kreps, X_both[i_source], X_both[1-i_source], X_tmp);
	  else
	      cump_reduce_kernel<false><<<dimGrid, dimBlock>>>(size, kreps, X_both[i_source], X_both[1-i_source], X_tmp);
	  if (verbose) {
	      mpf_array_set_cumpf(R, X_both[1-i_source], 1);
	      gmp_printf("  First element = %Fg\n", R[0]);
	      printf("  Kernel done.\n");
	  }
	  size = (size + threads-1)/threads;
	  i_source = 1-i_source;
      }
      cudaDeviceSynchronize();
      if (verbose)
	  printf("Getting result\n");
      mpf_array_set_cumpf(R, X_both[i_source], 1);
      mpf_set(dest, R[0]);
      double time = get_timer();
      //      printf("    time = %g\n", time);
      if (time < best_time)
	  best_time = time;
  }
  mpf_clear(R[0]);
  cumpf_array_clear (X_both[0]);
  cumpf_array_clear (X_both[1]);
  cumpf_array_clear (X_tmp);
  return best_time;
}

double  double_reduce (double *dval, bool product, int  n, int reps, int maxThreads, mpf_t  X[])
{
  double *Xd = (double *) calloc(n, sizeof(double));
  for (int i = 0; i < n; i++)
      Xd[i] = mpf_get_d(X[i]);

  double *X_both[2];
  checkCudaErrors(cudaMalloc((void **) &X_both[0], n * sizeof(double)));
  double *X_tmp;
  checkCudaErrors(cudaMalloc((void **) &X_tmp, n * sizeof(double)));
  
  *dval = product ? 1.0 : 0.0;
  
  int first_threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
  int first_blocks = (n + first_threads-1) / first_threads;
  checkCudaErrors(cudaMalloc((void **) &X_both[1], first_blocks * sizeof(double)));  
  double best_time = 1e9;
  for (int t = 0; t < trials; t++) {
      checkCudaErrors(cudaMemcpy(X_both[0], Xd, n*sizeof(double), cudaMemcpyHostToDevice));
      int size = n;
      bool first_time = true;
      int i_source = 0;
      cudaDeviceSynchronize();
      start_timer();
      while (first_time || size > 1) {
	  int kreps = first_time ? reps : 1;
	  first_time = false;
	  int threads = (size < maxThreads) ? nextPow2(size) : maxThreads;
	  int blocks = (size + threads-1)/threads;
	  dim3 dimGrid(blocks, 1, 1);
	  dim3 dimBlock(threads, 1, 1);
	  if (verbose)
	      printf("Running CUMP kernel.  Size = %d, Reps = %d\n", size, kreps);
	  if (product)
	      double_reduce_kernel<true><<<dimGrid, dimBlock>>>(size, kreps, X_both[i_source], X_both[1-i_source], X_tmp);
	  else
	      double_reduce_kernel<false><<<dimGrid, dimBlock>>>(size, kreps, X_both[i_source], X_both[1-i_source], X_tmp);
	  if (verbose)
	      printf("  Kernel done.\n");
	  size = (size + threads-1)/threads;
	  i_source = 1-i_source;
      }
      cudaDeviceSynchronize();
      if (verbose)
	  printf("Getting result\n");
      checkCudaErrors(cudaMemcpy(dval, X_both[i_source], 1 * sizeof(double), cudaMemcpyDeviceToHost));
      double time = get_timer();
      //      printf("    time = %g\n", time);
      if (time < best_time)
	  best_time = time;
  }
  free(Xd);
  cudaFree(X_both[0]);
  cudaFree(X_both[1]);
  cudaFree(X_tmp);
  return best_time;
}
