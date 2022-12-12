.PHONY : all

# CC = fcc
# C_FLAGS = -Kopenmp
# LD_FLAGS = -Kopenmp
CC = gcc
C_FLAGS = -fopenmp -D__NEON__
LD_FLAGS = -fopenmp

all: kernel_neon.o main.c
	$(CC) $(C_FLAGS) -c main.c -o main.o
	$(CC) $(LD_FLAGS) $(OBJS) main.o -o main 

kernel_neon.o: kernel_neon.s	
	$(CC) -c kernel_neon.s -o kernel_neon.o

clean:
	rm -rf main *.o