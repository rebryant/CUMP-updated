/*========================================================================
  Copyright (c) 2025 Randal E. Bryant, Carnegie Mellon University
  
  Permission is hereby granted, free of
  charge, to any person obtaining a copy of this software and
  associated documentation files (the "Software"), to deal in the
  Software without restriction, including without limitation the
  rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom
  the Software is furnished to do so, subject to the following
  conditions:
  
  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
  ========================================================================*/


/**** Functions that should be in the MPF distribution ****/
#include <gmp.h>

/* Raise base to pwr */
/* Assumes integer part of pwr fits into long */
void mpf_pow(mpf_t val, mpf_srcptr base, mpf_srcptr pwr);


/* 
   Create a string representation of an MPF value.
   In scientific notation
 */

const char *mpf_get_string(mpf_srcptr mval, int digits);

void mpf_settings_init(int prec, unsigned long int seed);

mpf_t *uniform_array_mpf(int len, double min, double max, double zpct, unsigned seed);

mpf_t *exponential_array_mpf(int len, double base, double min, double max, double zpct, unsigned seed);

double digit_precision(mpf_srcptr x_est, mpf_srcptr x_true);
