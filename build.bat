@echo off
setlocal
cd /d "%~dp0"

if not exist build mkdir build

g++ -std=c++14 -Wall -Wextra -Iinclude ^
  src/matrix.cpp src/activations.cpp src/neural_network.cpp src/dataset.cpp src/train_xor.cpp ^
  -o build/train_xor.exe

g++ -std=c++14 -Wall -Wextra -Iinclude ^
  src/matrix.cpp src/activations.cpp src/neural_network.cpp src/dataset.cpp src/train_mnist.cpp ^
  -o build/train_mnist.exe

echo Build complete.
