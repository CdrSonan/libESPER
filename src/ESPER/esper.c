#include "src/ESPER/esper.h"

#include "fftw/fftw3.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include "src/util.h"
#include "src/fft.h"
#include "src/ESPER/components.h"

//main function for ESPER audio analysis. Accepts a cSample as argument, and writes the results of the analysis back into the appropriate fields of the sample.
__declspec(dllexport) void __cdecl specCalc(cSample sample, engineCfg config)
{
    sample.config.batches = (sample.config.length / config.batchSize) + 1;
    fftwf_complex* buffer = stft(sample.waveform, sample.config.length, config);
    float* signalsAbs = (float*) malloc(sample.config.batches * (config.halfTripleBatchSize + 1) * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < sample.config.batches * (config.halfTripleBatchSize + 1); i++) {
        *(signalsAbs + i) = sqrtf(cpxAbsf(*(buffer + i)));
    }
    free(buffer);
    float* lowSpectra = lowRangeSmooth(sample, signalsAbs, config);
    float* highSpectra = highRangeSmooth(sample, signalsAbs, config);
    //finalizeSpectra(sample, lowSpectra, highSpectra, config);
    //free(lowSpectra);
    //free(highSpectra);
    //separateVoicedUnvoiced(sample, config);
    //averageSpectra(sample, config);
    for (int i = 0; i < sample.config.batches; i++) {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++) {
            *(sample.specharm + i * config.frameSize + config.nHarmonics + 2 + j) = *(highSpectra + i * (config.halfTripleBatchSize + 1) + j);
        }
    }
}
