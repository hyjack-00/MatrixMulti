.PHONY : all

# CC = fcc
# C_FLAGS = -Kopenmp
# LD_FLAGS = -Kopenmp
CC = gcc
C_FLAGS = -fopenmp -D__NEON__
LD_FLAGS = -fopenmp

OBJ += kernel_1.o
OBJ += kernel_2.o

all: $(OBJ) main.c
	$(CC) $(C_FLAGS) -c main.c -o main.o
	$(CC) $(LD_FLAGS) $(OBJ) main.o -o main 

kernel_1.o: kernel_1.s	
	$(CC) -c kernel_1.s -o kernel_1.o

kernel_2.o: kernel_2.s	
	$(CC) -c kernel_2.s -o kernel_2.o

clean:
	rm -rf main *.o