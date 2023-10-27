#pragma once

#include "fftw/fftw3.h"

//struct holding parameters related to data batching and spectral filtering shared across the entire engine.
//essentially a wrapper around relevant elements of global_consts.py
typedef struct
{
    unsigned int sampleRate;
    unsigned short tickRate;
    unsigned int batchSize;
    unsigned int tripleBatchSize;
    unsigned int halfTripleBatchSize; //exactly tripleBatchSize/2 without additional space for DC offset, since this is also used outside of rfft
    unsigned short filterBSMult;
    float DIOBias;
    float DIOBias2;
    float DIOTolerance;
    float DIOLastWinTolerance;
    unsigned short filterTEEMult;
    unsigned short filterHRSSMult;
    unsigned int nHarmonics;
    unsigned int halfHarmonics; //actually nHarmonics/2 + 1 to account for DC offset after rfft
    unsigned int frameSize; //nHarmonics + 2 for harmonic amplitudes and phases + halfTripleBatchSize + 1 for spectrum
    unsigned int ampContThreshold;
    unsigned int spectralRolloff1;
    unsigned int spectralRolloff2;
}
engineCfg;

//struct holding all sample-specific information and settings required by ESPER for a single audio sample
typedef struct
{
    unsigned int length;
    unsigned int batches;
    unsigned int pitchLength;
    unsigned int pitch;
    int isVoiced;
    int isPlosive;
    int useVariance;
    float expectedPitch;
    float searchRange;
    float voicedThrh;
    unsigned short specWidth;
    unsigned short specDepth;
    unsigned short tempWidth;
    unsigned short tempDepth;
}
cSampleCfg;

//struct holding an audio sample, and its settings
typedef struct
{
    float* waveform;
    int* pitchDeltas;
    float* specharm;
    float* avgSpecharm;
    float* excitation;
    cSampleCfg config;
}
cSample;

//struct containing all timing markers for a vocalSegment object. Used for resampling.
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

//struct for a dynamic integer array, usable for arbitrary purposes
typedef struct
{
    int* content;
    unsigned int length;
    unsigned int maxlen;
}
dynIntArray;

void dynIntArray_init(dynIntArray* array);

void dynIntArray_dealloc(dynIntArray* array);

void dynIntArray_append(dynIntArray* array, int value);

int ceildiv(int numerator, int denominator);

unsigned int findIndex(int* markers, unsigned int markerLength, int position);

unsigned int findIndex_double(double* markers, unsigned int markerLength, int position);

float cpxAbsf(fftwf_complex input);

float cpxArgf(fftwf_complex input);

float pi;
