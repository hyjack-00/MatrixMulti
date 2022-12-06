if [! -d "./build/"]; then
    mkdir ./build
    echo "Create ./build/"
fi
if [! -d "./bin/"]; then
    mkdir ./bin
    echo "Create ./bin/"
fi

cd ./build/
cmake ..
build
cd ..