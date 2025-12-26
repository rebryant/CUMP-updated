NVCC=nvcc
CPP=g++
CFLAGS=-O3 -g
DEST=home
DATA_CUMP = gpu-cump-$(DEST).csv
DATA_BIGNUM_CUMP = gpu-cump-bignum-$(DEST).csv
DATA_MATMUL_CUMP = gpu-cump-matmul-$(DEST).csv
INC = -I. -I.. -I/usr/local/cuda/include
CUMPLIB = ../libcump.a

all: eval_reduce matmul

.c.o:
	$(CPP) $(INC) $(CFLAGS) -c $< -o $@

eval_reduce: eval_reduce.cu clock.o mpf_extra.o
	$(NVCC) $(CFLAGS) -I . -I/usr/local/cuda/include -o eval_reduce eval_reduce.cu clock.o mpf_extra.o -lgmp $(CUMPLIB)

matmul: matmul.cu clock.o mpf_extra.o
	$(NVCC) $(CFLAGS) -I . -I/usr/local/cuda/include -o matmul matmul.cu clock.o mpf_extra.o -lgmp $(CUMPLIB)

run_er: eval_reduce
	./eval_reduce -n 1000000 -r 1000 -q -m 0.9999 -M 1.0001 > $(DATA_CUMP)     
	./eval_reduce -n 1000000 -r 1000 -q -m -1e308 -M 1e308 > $(DATA_BIGNUM_CUMP)     

run_mm: matmul
	./matmul -n 1000 -q -m 0 -M 1 > $(DATA_MATMUL_CUMP)     

clean:
	rm -f *.o *~ eval_reduce matmul
