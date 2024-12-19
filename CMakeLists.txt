cmake_minimum_required(VERSION 3.22)
project(libESPER VERSION 0.7.6 LANGUAGES C)

include(CTest)
enable_testing()

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

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
    target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw/.libs/libfftw3f.a")
    target_link_libraries(esper PRIVATE "${PROJECT_SOURCE_DIR}/nfft/.libs/libnfft3.a")
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
    target_include_directories(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw")
    target_include_directories(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/nfft")
    target_link_libraries(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/fftw/.libs/libfftw3f.a")
    target_link_libraries(test_esper PRIVATE "${PROJECT_SOURCE_DIR}/nfft/.libs/libnfft3.a")
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
    set_tests_properties(test_esper_${TEST_ARG} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_append:${PROJECT_SOURCE_DIR}/fftw_win")
    set_tests_properties(test_esper_${TEST_ARG} PROPERTIES ENVIRONMENT_MODIFICATION "PATH=path_list_append:${PROJECT_SOURCE_DIR}/nfft_win")
endforeach()
