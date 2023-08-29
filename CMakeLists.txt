cmake_minimum_required(VERSION 3.10)
project( ransac_test )

# Make flags
set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g" )
set( CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -gencode arch=compute_86,code=sm_86 -use_fast_math )

# Enable OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# Enable CUDA
find_package( CUDA REQUIRED )
include_directories( ${CUDA_INCLUDE_DIRS} )

# Enable PCL
find_package( PCL REQUIRED )
include_directories( ${PCL_INCLUDE_DIRS} )

# Add excutable
cuda_compile(CUDA_OBJECT src/ransac.cu)
set( SOURCES src/main.cpp src/utils.cpp )
add_executable( ${PROJECT_NAME} ${SOURCES} ${CUDA_OBJECT})

# Linking
target_link_libraries( ${PROJECT_NAME} 
    ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${PCL_LIBRARIES}
)