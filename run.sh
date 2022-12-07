#!/bin/bash

if [ -d ./build/ ]; then
    echo "./build/ exist"
else
    mkdir ./build
    echo "Create ./build/"
fi

if [ -d ./bin/ ]; then
    echo "./bin/ exist"
else
    mkdir ./bin
    echo "Create ./bin/"
fi

cd ./build/
cmake ..
make
cd ..
./bin/MatrixMulti
