cmake_minimum_required(VERSION 3.16)
project(NeuralNetworkFromScratch VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nn_core
    src/matrix.cpp
    src/activations.cpp
    src/neural_network.cpp
    src/dataset.cpp
)

target_include_directories(nn_core PUBLIC include)

add_executable(train_xor src/train_xor.cpp)
target_link_libraries(train_xor PRIVATE nn_core)

add_executable(train_mnist src/train_mnist.cpp)
target_link_libraries(train_mnist PRIVATE nn_core)

if(MSVC)
    target_compile_options(nn_core PRIVATE /W4)
else()
    target_compile_options(nn_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
