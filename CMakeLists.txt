cmake_minimum_required(VERSION 3.20)
project(libESPER VERSION 0.7.6 LANGUAGES C)

add_library(esper SHARED
    src/ESPER/esper.c
    src/ESPER/pitchCalcFallback.c
    src/ESPER/components.c
    src/fft.c
    src/interpolation.c
    src/util.c)
target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/")
target_compile_definitions(esper PRIVATE LIBESPER_BUILD)
set_property(TARGET esper PROPERTY C_STANDARD 11)
set_property(TARGET esper PROPERTY C_VISIBILITY_PRESET hidden)
if (WIN32)
    target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw_win/libfftw3f-3.lib")
else()
    target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw")
endif(WIN32)

add_library(resampler SHARED
    src/Resampler/resampler.c
    src/Resampler/loop.c
    src/interpolation.c
    src/util.c)
target_include_directories(resampler PRIVATE "${PROJECT_SOURCE_DIR}/")
target_compile_definitions(resampler PRIVATE LIBESPER_BUILD)
set_property(TARGET resampler PROPERTY C_STANDARD 11)
set_property(TARGET resampler PROPERTY C_VISIBILITY_PRESET hidden)

set_target_properties(esper resampler PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")

add_executable(test
    src/ESPER/esper.c
    src/ESPER/pitchCalcFallback.c
    src/ESPER/components.c
    src/Resampler/resampler.c
    src/Resampler/loop.c
    src/fft.c
    src/interpolation.c
    src/util.c
    src/test.c)
target_include_directories(test PRIVATE "${PROJECT_SOURCE_DIR}/")
target_compile_definitions(test PRIVATE LIBESPER_BUILD)
set_property(TARGET test PROPERTY C_STANDARD 11)
if (WIN32)
    target_link_libraries(test PRIVATE "${PROJECT_SOURCE_DIR}/fftw_win/libfftw3f-3.lib")
else()
    target_include_directories(test PRIVATE "${PROJECT_SOURCE_DIR}/fftw")
endif(WIN32)
add_test(NAME test_esper COMMAND test WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/")
