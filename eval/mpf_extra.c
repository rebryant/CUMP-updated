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

#include "mpf_extra.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>


/**** Functions that should be in the MPF distribution ****/

/* Raise base to pwr */
/* Assumes integer part of pwr fits into long */
void mpf_pow(mpf_t val, mpf_srcptr base, mpf_srcptr pwr) {
    /* For negative powers, will take reciprocal at end */
    mp_bitcnt_t prec = mpf_get_prec(val);
    bool is_neg = mpf_sgn(pwr) < 0;

    /* Absolute value of power */
    mpf_t apwr;
    mpf_init2(apwr, mpf_get_prec(pwr));
    if (is_neg)
	mpf_neg(apwr, pwr);
    else
	mpf_set(apwr, pwr);

    /* Take care of integer part */
    unsigned long ipwr = mpf_get_ui(apwr);
    mpf_pow_ui(val, base, ipwr);

    mpf_t frac;
    mpf_init2(frac, prec);
    mpf_sub_ui(frac, apwr, ipwr);

    mpf_t pbase;
    mpf_init2(pbase, prec);
    mpf_set(pbase, base);

    for (int i = 0; i < prec && mpf_sgn(frac) != 0 && mpf_cmp_d(pbase, 1.0) != 0; i++) {
	mpf_sqrt(pbase, pbase);
	mpf_mul_2exp(frac, frac, 1);
	unsigned long b = mpf_get_ui(frac);
	if (b > 0)
	    mpf_mul(val, val, pbase);
	mpf_sub_ui(frac, frac, b);
    }

    if (is_neg) {
	/* Reciprocal */
	mpf_set_d(frac, 1.0);
	mpf_div(val, frac, val);
    }
    mpf_clear(apwr);
    mpf_clear(frac);
    mpf_clear(pbase);
}

static const char *archive_string(const char *tstring) {
    char *rstring = (char *) malloc(strlen(tstring)+1);
    strcpy(rstring, tstring);
    return (const char *) rstring;
}

static void mpf_string(char *buf, mpf_srcptr val, int nsig) {
    char boffset = 0;
    mp_exp_t ecount;
    if (nsig <= 0)
	nsig = 1;
    if (nsig > 20)
	nsig = 20;
    char *sval = mpf_get_str(NULL, &ecount, 10, nsig, val);
    if (!sval || strlen(sval) == 0 || sval[0] == '0') {
	strcpy(buf, "0.0");
    } else {
	int voffset = 0;
	bool neg = sval[0] == '-';
	if (neg) {
	    voffset++;
	    buf[boffset++] = '-';
	}
	if (ecount == 0) {
	    buf[boffset++] = '0';
	    buf[boffset++] = '.';
	} else {
	    buf[boffset++] = sval[voffset++];
	    buf[boffset++] = '.';
	    ecount--;
	}
	if (sval[voffset] == 0)
	    buf[boffset++] = '0';
	else {
	    while(sval[voffset] != 0)
		buf[boffset++] = sval[voffset++];
	}
	if (ecount != 0) {
	    buf[boffset++] = 'e';
	    if (ecount > 0)
		buf[boffset++] = '+';
	    snprintf(&buf[boffset], 24, "%ld", (long) ecount);
	} else
	    buf[boffset] = 0;
    }
    free(sval);
}

const char *mpf_get_string(mpf_srcptr mval, int digits) {
    char buf[2048];
    mpf_string(buf, mval, digits);
    return archive_string(buf);
}

/***** Random number generation using MPF ****/

static gmp_randstate_t rstate;

void mpf_settings_init(int prec, unsigned long int seed) {
    mpf_set_default_prec(prec);
    gmp_randinit_default(rstate);
    gmp_randseed_ui(rstate, seed);
}

static void uniform_value_mpf(mpf_t r, double min, double max, double zpct) {
    double z = (double) random() / (double) ((1L<<31)-1);
    if (z * 100 < zpct) {
	mpf_set_d(r, 0.0);
	return;
    }
    mp_bitcnt_t prec = mpf_get_prec(r);
    mpf_urandomb(r, rstate, prec);
    mpf_t scale;
    mpf_init2(scale, prec);
    mpf_set_d(scale, max-min);
    mpf_mul(r, r, scale);
    mpf_set_d(scale, min);
    mpf_add(r, r, scale);
    mpf_clear(scale);
}

static void exponential_value_mpf(mpf_t r, double base, double minp, double maxp, double zpct) {
    double z = (double) random() / (double) ((1L<<31)-1);
    if (z * 100 < zpct)
	mpf_set_d(r, 0.0);
    mp_bitcnt_t prec = mpf_get_prec(r);
    mpf_t pwr;
    mpf_init2(pwr, prec);
    uniform_value_mpf(pwr, minp, maxp, 0.0);
    mpf_t mbase;
    mpf_init2(mbase, prec);
    mpf_set_d(mbase, base);
    mpf_pow(r, mbase, pwr);
    mpf_clear(pwr);
    mpf_clear(mbase);
}

mpf_t *uniform_array_mpf(int len, double min, double max, double zpct, unsigned seed) {
    srandom(seed);
    mpf_t *mval = (mpf_t *) calloc(len, sizeof(mpf_t));
    for (int i = 0; i < len; i++) {
	mpf_init(mval[i]);
	uniform_value_mpf(mval[i], min, max, zpct);
    }
    return mval;
}

mpf_t *exponential_array_mpf(int len, double base, double min, double max, double zpct, unsigned seed) {
    srandom(seed);
    mpf_t *mval = (mpf_t *) calloc(len, sizeof(mpf_t));
    for (int i = 0; i < len; i++) {
	mpf_init(mval[i]);
	exponential_value_mpf(mval[i], base, min, max, zpct);
    }
    return mval;
}



double digit_precision(mpf_srcptr x_est, mpf_srcptr x_true) {
    if (mpf_cmp(x_est, x_true) == 0)
	return 1e6;
    if (mpf_cmp_d(x_est, 0.0) == 0 || mpf_cmp_d(x_true, 0.0) == 0)
	return 0.0;
    mpf_t rel;
    mpf_init2(rel, mpf_get_prec(x_true) * 2);
    mpf_sub(rel, x_est, x_true);
    mpf_div(rel, rel, x_true);
    mpf_abs(rel, rel);
    long int exp;
    double drel = mpf_get_d_2exp(&exp, rel);
    double dp = -(log10(drel) + log10(2.0) * exp);
    mpf_clear(rel);
    return dp < 0 ? 0 : dp;
}
