cmake_minimum_required(VERSION 3.22)
project(libESPER VERSION 1.0.0 LANGUAGES C)

include(ExternalProject)

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
        PREFIX "${PROJECT_SOURCE_DIR}/fftwf"
        URL "http://www.fftw.org/fftw-3.3.10.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP true
        CONFIGURE_COMMAND "${PROJECT_SOURCE_DIR}/fftwf/src/fftwf/configure" --enable-float --enable-shared=yes --enable-static=no
        INSTALL_COMMAND ""
    )
    ExternalProject_add(fftw
        PREFIX "${PROJECT_SOURCE_DIR}/fftw"
        URL "http://www.fftw.org/fftw-3.3.10.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP true
        CONFIGURE_COMMAND "${PROJECT_SOURCE_DIR}/fftw/src/fftw/configure" --enable-threads --with-combined-threads --enable-shared=yes --enable-static=no
        INSTALL_COMMAND ""
    )
    ExternalProject_add(nfft
        PREFIX "${PROJECT_SOURCE_DIR}/nfft"
        URL "https://www-user.tu-chemnitz.de/~potts/nfft/download/nfft-3.5.3.tar.gz"
        DOWNLOAD_EXTRACT_TIMESTAMP true
        CONFIGURE_COMMAND "${PROJECT_SOURCE_DIR}/nfft/src/nfft/configure" --enable-all --with-fftw3-libdir=${PROJECT_SOURCE_DIR}/fftw/src/fftw-build/.libs/ --enable-shared=yes --enable-static=no
        INSTALL_COMMAND ""
    )
    add_dependencies(nfft fftw)
    add_dependencies(esper fftwf nfft)
    target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/fftwf/src/fftwf")
    target_include_directories(esper PRIVATE "${PROJECT_SOURCE_DIR}/nfft/src/nfft")
    target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/fftwf/src/fftwf-build/.libs/libfftw3f.so")
    target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/nfft/src/nfft-build/.libs/libnfft3.so")
    target_link_libraries(esper PRIVATE m)
    set_target_properties(esper PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/")
    add_custom_command(TARGET esper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:esper> "${CMAKE_BINARY_DIR}/bundled/libesper.so"
    )
    add_custom_command(TARGET esper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/fftwf/src/fftwf-build/.libs/libfftw3f.so.3" "${CMAKE_BINARY_DIR}/bundled/libfftw3f.so.3"
    )
    add_custom_command(TARGET esper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/fftw/src/fftw-build/.libs/libfftw3.so.3" "${CMAKE_BINARY_DIR}/bundled/libfftw3.so.3"
    )
    add_custom_command(TARGET esper POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${PROJECT_SOURCE_DIR}/nfft/src/nfft-build/.libs/libnfft3.so.4" "${CMAKE_BINARY_DIR}/bundled/libnfft3.so.4"
        COMMAND patchelf --set-rpath '\$\$ORIGIN/' "${CMAKE_BINARY_DIR}/bundled/libnfft3.so.4"
    )
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
        #set_tests_properties(test_esper_${TEST_ARG} PROPERTIES ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_append:${PROJECT_SOURCE_DIR}/fftwf/src/fftwf/.libs")
        #set_tests_properties(test_esper_${TEST_ARG} PROPERTIES ENVIRONMENT_MODIFICATION "LD_LIBRARY_PATH=path_list_append:${PROJECT_SOURCE_DIR}/nfft/src/nfft/.libs")
    endif(WIN32)
endforeach()
