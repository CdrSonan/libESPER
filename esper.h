//Copyright 2024 Johannes Klatt

//This file is part of libESPER.
//libESPER is free software: you can redistribute it and /or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
//libESPER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with Nova - Vox.If not, see <https://www.gnu.org/licenses/>.

#pragma once


//This header file contains the interface for the libESPER library.
//It is designed to be used when linking against a prebuilt dll of the library, NOT when building the library itself.

#ifdef LIBESPER_BUILD
    #ifdef _WIN32
        #define LIBESPER_EXPORT __declspec(dllexport)
        #define LIBESPER_CDECL __cdecl
    #elif __GNUC__ >= 4
        #define LIBESPER_EXPORT __attribute__((visibility("default")))
        #define LIBESPER_CDECL __attribute__((__cdecl__))
    #else
        #define LIBESPER_EXPORT
        #define LIBESPER_CDECL
    #endif
#else
    #ifdef _WIN32
        #define LIBESPER_EXPORT __declspec(dllimport)
        #define LIBESPER_CDECL __cdecl
    #else
        #define LIBESPER_EXPORT
        #define LIBESPER_CDECL
    #endif
#endif

/*
struct holding all settings used by ESPER as a whole.
These settings should stay the same when combining multiple samples,
as they are used to determine the size of several internal arrays.

members:
sampleRate: the sample rate of the audio samples being processed in Hz.
tickRate: the number of processing windows per second of audio. Should be as close to sampleRate/batchSize as possible.
batchSize: the number of samples per processing window. Numbers with low prime factors improve performance.
tripleBatchSize: HAS to be exactly 3 times batchSize. The lowest frequency the engine can properly process is given by sampleRate/tripleBatchSize.
halfTripleBatchSize: HAS to be exactly tripleBatchSize/2.
nHarmonics: the number of voiced harmonics to be extracted from a sample.
halfHarmonics: HAS to be exactly nHarmonics/2 + 1.
frameSize: HAS to be exactly nHarmonics + halfTripleBatchSize + 3.
breCompPremul: the pre-multiplier applied to the compensation term used with positive breathiness values. Higher values increase the volume of unvoiced sounds when using positive breathiness.
*/
typedef struct
{
    unsigned int sampleRate;
    unsigned short tickRate;
    unsigned int batchSize;
    unsigned int tripleBatchSize;
    unsigned int halfTripleBatchSize;
    unsigned int nHarmonics;
    unsigned int halfHarmonics; //expected to be nHarmonics/2 + 1 (to account for DC offset after rfft)
    unsigned int frameSize; //expected to be nHarmonics + halfTripleBatchSize + 3 for joint harmonics + spectrum representation
    float breCompPremul;
}
engineCfg;

/*
struct holding all sample-specific information and settings required by ESPER for a single audio sample.

members:
length: the length of the audio sample in samples.
batches: the length of the audio sample in batches. Should be floor(length/batchSize).
pitchLength: the length of the pitchDeltas array of the sample. Should be roughly equal to batches.
markerLength: the length of the pitchMarkers array of the sample.
pitch: the pitch of the sample as a wavelength in samples. Should be the average of the pitchDeltas array.
isVoiced: whether the sample is voiced or unvoiced. 1 for voiced, 0 for unvoiced.
isPlosive: whether the sample is a plosive sound. 1 for plosive, 0 for non-plosive. Only used in Nova-Vox, no effect in any ESPER functions.
useVariance: whether to apply an additional postprocessing step to dampen spectrum and harmonic outliers. 1 if yes, 0 if no.
expectedPitch: the expected pitch of the sample in Hz.
searchRange: the range around the expected pitch in which to search for pitch markers. The range is determined as [expectedPitch*searchRange, expectedPitch/searchRange].
tempWidth: filter width used for temporal smoothing between spectra.
*/
typedef struct
{
    unsigned int length;
    unsigned int batches;
    unsigned int pitchLength;
    unsigned int markerLength;
    unsigned int pitch;
    int isVoiced;
    int isPlosive;
    int useVariance;
    float expectedPitch;
    float searchRange;
    unsigned short tempWidth;
}
cSampleCfg;

/*
struct holding an audio sample and its settings.

members:
waveform: the audio sample as an array of floats.
pitchDeltas: an array describing the pitch of the sample as wavelengths measured as a number of samples. Each element should roughly correspond to the pitch of a processing window.
pitchMarkers: an array of pitch markers used to determine the pitch of the sample. Each element marks a border between two pitch periods.
specharm: This array combines three sections for each processing window:
- the deviation of the harmonic amplitudes from their average
- the phase of the harmonics, relative to the f0 phase
- the deviation of the average fourier coefficients of the unvoiced part of the sample from the average of the entire sample.
avgSpecharm: the average harmonic amplitudes and spectrum of the sample.
config: the configuration object for the sample.
*/
typedef struct
{
    float* waveform;
    int* pitchDeltas;
    int* pitchMarkers;
    char* pitchMarkerValidity;
    float* specharm;
    float* avgSpecharm;
    cSampleCfg config;
}
cSample;

/*
struct containing all timing markers for a vocalSegment object. Used for resampling.

members:
start1, start2, start3: three timing markers representing the transition from the previous sample to this sample.
end1, end2, end3: three timing markers representing the transition from this sample to the next one.
windowStart: start marker used for audio output.
windowEnd: end marker used for audio output.
offset: an offset applied to the voiced part of the sample before looping.
*/
typedef struct
{
    unsigned int start1;
    unsigned int start2;
    unsigned int start3;
    unsigned int end1;
    unsigned int end2;
    unsigned int end3;
    unsigned int windowStart;
    unsigned int windowEnd;
    unsigned int offset;
}
segmentTiming;



//ANALYSIS FUNCTIONS

//Given a sample and its configuration, this function determines the pitch and fills the pitchMarkers and pitchDeltas arrays and related fields.
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL pitchCalcFallback(cSample* sample, engineCfg config);

//Main function of this library. Given a sample and its configuration, this function determines its voiced harmonics and unvoiced residuals, and writes them to the appropriate fields.
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL specCalc(cSample sample, engineCfg config);



//RESAMPLING FUNCTIONS

/*Given a specharm array, this function resamples it to the desired length and writes the result to the output array.
  Arguments:
    avgSpecharm: the average specharm array of the sample.
    specharm: the specharm deviation array to be resampled.
    length: the length of the specharm array.
    steadiness: an array of values representing the steadiness of the sample. Higher values indicate lower contribution of the deviations.
    spacing: the spacing between individual sample instances used when looping.
    startCap: whether to apply a "cap" (resample from start1 marker and fade in sample) at the start, or perform default resampling (resample from start3 marker).
    endCap: whether to apply a "cap" (resample to end3 marker and fade out sample) at the end, or perform default resampling (resample to end1 marker).
    output: the array to write the resampled specharm to. Should already be allocated.
    timings: the timing settings for the sample.
    config: the engine configuration settings used.
*/
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL resampleSpecharm(float* avgSpecharm, float* specharm, int length, float* steadiness, float spacing, int startCap, int endCap, float* output, segmentTiming timings, engineCfg config);


/*Given a pitchDeltas array, this function resamples it to the desired length and writes the result to the output array.
  Arguments:
    pitchDeltas: the pitchDeltas array to be resampled.
    length: the length of the pitchDeltas array.
    pitch: the average pitch of the sample as a wavelength, measured in sample points.
    spacing: the spacing between individual sample instances used when looping.
    startCap: whether to apply a "cap" (resample from start1 marker and fade in sample) at the start, or perform default resampling (resample from start3 marker).
    endCap: whether to apply a "cap" (resample to end3 marker and fade out sample) at the end, or perform default resampling (resample to end1 marker).
    output: the array to write the resampled pitchDeltas to. Should already be allocated.
    requiredSize: the length of the output array.
    timings: the timing settings for the sample.
*/
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL resamplePitch(int* pitchDeltas, int length, float pitch, float spacing, int startCap, int endCap, float* output, int requiredSize, segmentTiming timings);



//MODIFICATION FUNCTIONS

//Given a specharm array and its length, this function applies a breathiness modification to it.
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL applyBreathiness(float* specharm, float* breathiness, int length, engineCfg config);

/*This function applies a pitch shift to a specharm array.
  Arguments:
    specharm: the specharm array to be modified.
    srcPitch: the pitch of the source sample as a wavelength, measured in sample points.
    tgtPitch: the target pitch to shift to as a wavelength, measured in sample points.
    formantShift: additional shift applied to the formants of the sample, before performing the main pitch shift.
    breathiness: the breathiness curve the sample. Higher values increase the pitch shift of unvoiced sounds.
    length: the length of the specharm array.
    config: the engine configuration settings used.
*/
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL pitchShift(float* specharm, float* srcPitch, float* tgtPitch, float* formantShift, float* breathiness, int length, engineCfg config);

//applies a dynamics effect to a specharm array. Also requires a pitch array.
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL applyDynamics(float* specharm, float* dynamics, float* pitch, int length, engineCfg config);

//applies a brightness effectto a specharm array.
extern "C" void LIBESPER_CDECL applyBrightness(float* specharm, float* brightness, int length, engineCfg config);

//applies a growl effect to a specharm array. lfophase is a pointer to a single float value representing the phase of the growl effect at the beginning of the sample. It will be updated to the phase at the end of the sample.
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL applyGrowl(float* specharm, float* growl, float* lfoPhase, int length, engineCfg config);

//applies a roughness effect to a specharm array.
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL applyRoughness(float* specharm, float* roughness, int length, engineCfg config);



//RENDERING FUNCTIONS

//Given a specharm array, this function renders the unvoiced portion of the sample into the target array.
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL renderUnvoiced(float* specharm, float* target, int length, engineCfg config);

/*Render the voiced portion of the sample into the target array.
  Arguments:
    specharm: the specharm array of the sample.
    pitch: the pitch of the sample as a wavelength, measured in sample points.
    phase: pointer to a single float value representing the phase at the beginning of the sample. Will be updated to the phase at the end of the sample, which can be used to align consecutive samples.
    target: the array to write the rendered sample to. Should already be allocated and initialized. Unlike in renderUnvoiced, the voiced component is added to the array, rather than overwriting it.
    length: the length of the target array.
    config: the engine configuration settings used.
*/
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL renderVoiced(float* specharm, float* pitch, float* phase, float* target, int length, engineCfg config);

//Function combining renderUnvoiced and renderVoiced to render an entire sample.
extern "C" LIBESPER_EXPORT void LIBESPER_CDECL render(float* specharm, float* pitch, float* phase, float* target, int length, engineCfg config);
