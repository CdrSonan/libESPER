cmake_minimum_required(VERSION 3.20)
project(libESPER VERSION 0.7.6 LANGUAGES C)

enable_testing()

add_library(esper SHARED
    src/ESPER/esper.c
    src/ESPER/pitchCalcFallback.c
    src/ESPER/components.c
    src/Resampler/resampler.c
    src/Resampler/loop.c
    src/Renderer/renderer.c
    src/fft.c
    src/interpolation.c
    src/util.c)
target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/")
target_compile_definitions(esper PRIVATE LIBESPER_BUILD)
set_property(TARGET esper PROPERTY C_STANDARD 11)
set_property(TARGET esper PROPERTY C_VISIBILITY_PRESET hidden)
if (WIN32)
    target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw_win/libfftw3f-3.lib")
    target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/nfft_win/libnfft3_threads-4.lib")
else()
    target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw")
    target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/nfft")
endif(WIN32)

add_executable(test_esper
    src/ESPER/esper.c
    src/ESPER/pitchCalcFallback.c
    src/ESPER/components.c
    src/Resampler/resampler.c
    src/Resampler/loop.c
    src/Renderer/renderer.c
    src/fft.c
    src/interpolation.c
    src/util.c
    src/test.c)
target_include_directories(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/")
target_compile_definitions(test_esper PRIVATE LIBESPER_BUILD)
set_property(TARGET test_esper PROPERTY C_STANDARD 11)
add_test(NAME test_esper COMMAND test_esper WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/")

set_target_properties(esper test_esper PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/build")

if (WIN32)
    target_link_libraries(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw_win/libfftw3f-3.lib")
    SET_TESTS_PROPERTIES(test_esper PROPERTIES ENVIRONMENT "${PROJECT_SOURCE_DIR}/fftw_win;$ENV{PATH}")
    add_custom_command(TARGET test_esper POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/fftw_win/libfftw3f-3.dll" "${PROJECT_SOURCE_DIR}/build/release/libfftw3f-3.dll")
else()
    target_include_directories(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw")
    SET_TESTS_PROPERTIES( test_esper PROPERTIES ENVIRONMENT "PATH=${PROJECT_SOURCE_DIR}/fftw;$ENV{PATH}")
    add_custom_command(TARGET test_esper POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/fftw/libfftw3f-3.so" "${PROJECT_SOURCE_DIR}/build/release/libfftw3f-3.so")
endif(WIN32)

