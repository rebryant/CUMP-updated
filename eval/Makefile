NVCC=nvcc
CPP=g++
CFLAGS=-O3 -g
DEST=home
DATA_CUMP = gpu-cump-$(DEST).txt
INC = -I. -I.. -I/usr/local/cuda/include
CUMPLIB = ../libcump.a

all: eval_reduce

.c.o:
	$(CPP) $(INC) $(CFLAGS) -c $< -o $@

eval_reduce: eval_reduce.cu clock.o
	$(NVCC) $(CFLAGS) -I . -I/usr/local/cuda/include -o eval_reduce eval_reduce.cu clock.o -lgmp $(CUMPLIB)

run: eval_reduce
	./eval_reduce -n 1000000 -r 1000 -q -m 0.9999 -M 1.0001 > $(DATA_CUMP)     

clean:
	rm -f *.o *~ eval_reduce
