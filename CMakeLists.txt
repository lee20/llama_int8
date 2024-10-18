cmake_minimum_required(VERSION 3.18)

project(llama2cpp)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -D WITH_OPENMP=ON -fopenmp")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
add_compile_options(-O3 -mavx2 -mfma)
set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON)

set(TOKENZIER_CPP_PATH ./tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
file(GLOB SRC_FILES "${SRC_DIR}/*.cpp")
add_executable(main ${SRC_FILES})

target_include_directories(main PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_include_directories(main PRIVATE ./include)

target_link_libraries(main PRIVATE tokenizers_cpp)
