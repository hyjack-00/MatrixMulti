gcc -fopenmp -c neon_cal.s -o neon_cal.o
gcc -fopenmp -c main.c     -o main.o

gcc -fopenmp neon_cal.o main.o -o main