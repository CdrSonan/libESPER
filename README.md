# libESPER

C implementation of the ESPER framework for vocal/singing analysis and resampling. Developed for Nova-Vox.

# Features

At its core, ESPER is a way to parametrize spoken or sung audio as the sum of a weakly periodic voiced and aperiodic unvoiced part.
This representation maintains high audio quality, is usable as an AI feature descriptor, can be used for heavy modification using mathematical methods without a loss in perceived quality, and can be transformed back to an audio waveform relatively quickly.
This comes at the cost of a slow "forward" transform and relatively large size of the parametrization.
The libESPER library builds on this concept, and bundles it with the features required by a typical, concatenative vocal synthesis engine:
- Graph-based pitch detection algorithm (Outputs both pitch-synchronous markers and time-synchronous pitch)
- Forward and reverse ESPER transform
- Resampling functions for changing the length of audio clips and other types of data
- Pseudo-spectral pitch shifting function
- Various vocal effects

# Future work
Currently, there are three features planned for future versions of the library:
- Fast approximate forward transform: This would be an alternate forward transform trading some audio quality for vastly increased speed. Such a system would allow ESPER to be used in AI-based speech modification, dictation or control systems, or any other scenario where forward processing speed is more important than perfectly maintaining audio quality
- Data compression: The current ESPER representation still contains a large amount of redundant data. However, by applying specifically tailored compression methods to it, it should be possible to bring the representation size down to a lefel similar to the widely used Mel-Space Cepstral Coefficients. This would further improve the usability of ESPER as a base for various AI systems.
- Guided pitch detection: Currently, the pitch detection algorithm can only process a single (optional) "expected pitch" value for the whole sample. It is planned to eventually expand on this, and offer support for approximate pitch curves as input for the algorithm, effectively turning it into a two-stage setup, with both better accuracy and speed than the current version.

# Building

## Windows

To build libESPER on Windows, you will need the following software:
- Microsoft Visual Studio 2019 or later (earlier versions may work as well, but have not been tested)
- CMake version 22 or later

Start by cloning the repository into a folder of your choice, or by downloading and extracting the .zip file of the source code.
Within the directory of the repository, create a new folder named "build" (or a different name of your choice).
Start CMake GUI and select the main directory of the repository (the folder containing CMakeLists.txt) as the source folder, and the newly created build folder as the output directory.
Then run CMake.
This will create a .sln Visual Studio solution file in the build folder, among several other files.
Open the .sln file with Visual Studio, and find the project named "esper" in the project explorer on the right. Right click it, then select "build project" to compile the library.
After compilation, the resulting library (esper.dll) and its dependencies will be placed in build/bundled.

## Linux

All packages required for building libESPER on Linux are included in the buildessentials metapackage.
After installing it, clone the repository using the git clone command. Navigate into the new directory, and create a new folder named "build" inside it.
Navigate into the build folder, and run the command
cmake ..
followed by
make
This will download and build the required dependencies, and compile libESPER itself.
Like on Windows, the resulting .so files are placed into ./build/bundled.
Because the dependencies of libfftw and libnfft are compiled on Linux, rather than being already provided as binaries like on Windows, the build process takes significantly longer.

# Usage

The interface used for integrating libESPER into another program is provided in the esper.h file in the repository root folder.
On Linux, you can use this header to directly link against the built .so file.
On Windows, you instead need to link against the corresponding .lib file. It can be found in build/Debug or build/Release, depending on your chosen build configuration.
When using the Debug configuration, the folder also contains the .pdb file required for debugger compatibility.

When bundling libESPER with another program, it is important to place all .dll/.so files in a location where they can be found by the main executable and each other.
On Windows, this means they either need to be placed in the same folder as the .exe file using them, or in a location included in the PATH environment variable.
On Linux, all .so files are set up to mimic this behavior. Their rpath attribute is configured so they search for their dependencies in the same folder, and in addition, dependencies can be found in all locations specified by the runpath environment variable.
However, it remains the responsibility of the programmer to ensure the calling executable finds libESPER.so.
