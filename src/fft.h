#pragma once

#include "src/util.h"
#include "fftw/fftw3.h"

fftwf_complex* fft(fftwf_complex* input, int length);

fftwf_complex* ifft(fftwf_complex* input, int length);

fftwf_complex* rfft(float* input, int length);

void rfft_inpl(float* input, int length, fftwf_complex* output);

float* irfft(fftwf_complex* input, int length);

void irfft_inpl(fftwf_complex* input, int length, float* output);

fftwf_complex* stft(float* input, int length, engineCfg config);

void stft_inpl(float* input, int length, engineCfg config, float* output);

float* istft(fftwf_complex* input, int batches, int targetLength, engineCfg config);

float* istft_hann(fftwf_complex* input, int batches, int targetLength, engineCfg config);
