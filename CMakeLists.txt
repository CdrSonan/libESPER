cmake_minimum_required(VERSION 3.22)
project(libESPER VERSION 0.7.6 LANGUAGES C)

include(CTest)
enable_testing()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/bundled")

add_library(esper SHARED
    src/ESPER/esper.c
    src/ESPER/pitchCalcFallback.c
    src/ESPER/components.c
    src/Resampler/resampler.c
    src/Resampler/loop.c
    src/Renderer/renderer.c
    src/Renderer/modifiers.c
    src/fft.c
    src/interpolation.c
    src/util.c
    esper.def)
target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/")
target_compile_definitions(esper PRIVATE LIBESPER_BUILD)
set_property(TARGET esper PROPERTY C_STANDARD 17)
set_property(TARGET esper PROPERTY C_VISIBILITY_PRESET hidden)
set_property(TARGET esper PROPERTY OUTPUT_NAME "esper")
set_property(TARGET esper PROPERTY WINDOWS_EXPORT_ALL_SYMBOLS ON)
if (WIN32)
    target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw_win/libfftw3f-3.lib")
    target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/nfft_win/libnfft3_threads-4.lib")
    add_custom_command(TARGET esper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:esper> "${CMAKE_BINARY_DIR}/bundled/esper.dll"
    )
    add_custom_command(TARGET esper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/fftw_win/libfftw3f-3.dll" "${CMAKE_BINARY_DIR}/bundled/libfftw3f-3.dll"
    )
    add_custom_command(TARGET esper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/nfft_win/libnfft3_threads-4.dll" "${CMAKE_BINARY_DIR}/bundled/libnfft3_threads-4.dll"
    )
else()
    ExternalProject_add(fftwf
        PREFIX "${CMAKE_BINARY_DIR}/fftwf"
        URL "http://www.fftw.org/fftw-3.3.10.tar.gz"
        CONFIGURE_COMMAND "${CMAKE_BINARY_DIR}/fftwf/src/fftwf/configure" --enable-float --disable-fortran --enable-shared=no --enable-static=yes
    )
    ExternalProject_add(fftw
        PREFIX "${CMAKE_BINARY_DIR}/fftw"
        URL "http://www.fftw.org/fftw-3.3.10.tar.gz"
        CONFIGURE_COMMAND "${CMAKE_BINARY_DIR}/fftw/src/fftw/configure" --enable-threads --enable-shared=no --enable-static=yes
    )
    ExternalProject_add(nfft
        PREFIX "${CMAKE_BINARY_DIR}/nfft"
        URL "https://www-user.tu-chemnitz.de/~potts/nfft/download/nfft-3.5.3.tar.gz"
        CONFIGURE_COMMAND "${CMAKE_BINARY_DIR}/nfft/src/nfft/configure" --enable-all --enable-openmp --enable-shared=no --enable-static=yes
    )
    add_dependencies(nfft fftw)
    add_dependencies(esper fftwf nfft)
    target_include_directories(esper PRIVATE "${CMAKE_BINARY_DIR}/fftwf/src/fftwf")
    target_include_directories(esper PRIVATE "${CMAKE_BINARY_DIR}/nfft/src/nfft")
    target_link_libraries(esper PRIVATE "${CMAKE_BINARY_DIR}/fftwf/src/fftwf-build/.libs/libfftw3f.a")
    target_link_libraries(esper PRIVATE "${CMAKE_BINARY_DIR}/nfft/src/nfft-build/.libs/libnfft3.a")
    target_link_libraries(esper PRIVATE m)
endif(WIN32)

add_executable(test_esper
    src/test.c)
target_include_directories(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/")
target_compile_definitions(test_esper PRIVATE LIBESPER_BUILD)
set_property(TARGET test_esper PROPERTY C_STANDARD 17)

target_link_libraries(test_esper PRIVATE esper)
if (WIN32)
    target_link_libraries(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw_win/libfftw3f-3.lib")
    target_link_libraries(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/nfft_win/libnfft3_threads-4.lib")
else()
    add_dependencies(test_esper fftwf nfft esper)
    target_include_directories(test_esper PRIVATE "${CMAKE_BINARY_DIR}/fftwf/src/fftwf")
    target_include_directories(test_esper PRIVATE "${CMAKE_BINARY_DIR}/nfft/src/nfft")
    target_link_libraries(test_esper PRIVATE "${CMAKE_BINARY_DIR}/fftwf/src/fftwf-build/.libs/libfftw3f.a")
    target_link_libraries(test_esper PRIVATE "${CMAKE_BINARY_DIR}/nfft/src/nfft-build/.libs/libnfft3.a")
    target_link_libraries(test_esper PRIVATE m)
endif(WIN32)

# Add multiple tests with different arguments
set(TEST_ARGS_LIST
    "arg1"
    "arg2"
    "arg3"
    "arg4"
)

foreach(TEST_ARG IN LISTS TEST_ARGS_LIST)
    add_test(NAME test_esper_${TEST_ARG} COMMAND test_esper ${TEST_ARG} WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/")
    if (WIN32)
        set_tests_properties(test_esper_${TEST_ARG} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_append:${PROJECT_SOURCE_DIR}/fftw_win")
        set_tests_properties(test_esper_${TEST_ARG} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_append:${PROJECT_SOURCE_DIR}/nfft_win")
    else()
        #This should not be necessary with static linking
        set_tests_properties(test_esper_${TEST_ARG} PROPERTIES ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_append:${PROJECT_SOURCE_DIR}/fftwf/src/fftwf/.libs")
        set_tests_properties(test_esper_${TEST_ARG} PROPERTIES ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_append:${PROJECT_SOURCE_DIR}/nfft/src/nfft/.libs")
    endif(WIN32)
endforeach()
