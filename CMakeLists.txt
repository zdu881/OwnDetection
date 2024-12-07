cmake_minimum_required(VERSION 3.10)

# Set the project name5
project(DLdetection)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR}/includes)
# Find OpenCV package
find_package(OpenCV REQUIRED)

# Add the executable
add_executable(DLdetection src/detectionv.cpp )

# Link OpenCV libraries
target_link_libraries(DLdetection ${OpenCV_LIBS})
