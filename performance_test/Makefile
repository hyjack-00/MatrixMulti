.PHONY : all

CC = gcc
C_FLAGS = -fopenmp -O2
LD_FLAGS = -fopenmp

KERNELS = $(wildcard *.s)
OBJS = $(KERNELS:%.s=%.o)

all: $(OBJS) main.c
	$(CC) $(C_FLAGS) -c main.c -o main.o
	$(CC) $(LD_FLAGS) $(OBJS) main.o -o main 

%.o: %.s
	$(CC) -c $< -o $@

clean:
	rm -rf main *.o