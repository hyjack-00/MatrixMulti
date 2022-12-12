.PHONY : all

# CC = fcc
# C_FLAGS = -Kopenmp
# LD_FLAGS = -Kopenmp
CC = gcc
C_FLAGS = -fopenmp -D__NEON__
LD_FLAGS = -fopenmp

OBJ += kernel_neon_fp32.o


all: $(OBJ) main.c
	$(CC) $(C_FLAGS) -c main.c -o main.o
	$(CC) $(LD_FLAGS) $(OBJ) main.o -o main 

kernel_neon_fp32.o: kernel_neon_fp32.s	
	$(CC) -c kernel_neon_fp32.s -o kernel_neon_fp32.o

clean:
	rm -rf main *.o