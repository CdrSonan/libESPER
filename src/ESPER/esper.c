//Copyright 2023 Johannes Klatt

//This file is part of libESPER.
//libESPER is free software: you can redistribute it and /or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
//libESPER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with Nova - Vox.If not, see <https://www.gnu.org/licenses/>.

#include "src/ESPER/esper.h"

#include "fftw/fftw3.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include "src/util.h"
#include "src/fft.h"
#include "src/ESPER/components.h"

//main function for ESPER audio analysis. Accepts a cSample as argument, and writes the results of the analysis back into the appropriate fields of the sample.
void LIBESPER_CDECL specCalc(cSample sample, engineCfg config)
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
    finalizeSpectra(sample, lowSpectra, highSpectra, config);
    free(lowSpectra);
    free(highSpectra);
    separateVoicedUnvoiced(sample, config);
    averageSpectra(sample, config);
}
