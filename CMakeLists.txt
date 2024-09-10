cmake_minimum_required(VERSION 3.20)
project(colour)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCE_FILES main.cpp)
add_executable(colour ${SOURCE_FILES})
include_directories(/usr/local/include)

find_package(SFML 2.6 COMPONENTS system window graphics network audio REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${SFML_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(colour sfml-system sfml-window sfml-graphics sfml-audio sfml-network ${OpenCV_LIBS})