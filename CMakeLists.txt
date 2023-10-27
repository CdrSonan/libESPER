cmake_minimum_required(VERSION 3.20)
project(libESPER VERSION 0.7.6 LANGUAGES C)

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(esper SHARED
    src/ESPER/esper.c
    src/ESPER/pitchCalcFallback.c
    src/ESPER/components.c
    src/fft.c
    src/interpolation.c
    src/util.c)
target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/")
set_property(TARGET esper PROPERTY C_STANDARD 11)
target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw/libfftw3f-3.lib")

add_library(resampler SHARED
    src/Resampler/resampler.c
    src/Resampler/loop.c
    src/interpolation.c
    src/util.c)
target_include_directories(resampler PRIVATE "${PROJECT_SOURCE_DIR}/")
set_property(TARGET resampler PROPERTY C_STANDARD 11)

#set_target_properties(esper resampler PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")
