//Copyright 2023 Johannes Klatt

//This file is part of libESPER.
//libESPER is free software: you can redistribute it and /or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
//libESPER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with Nova - Vox.If not, see <https://www.gnu.org/licenses/>.

#include "src/Resampler/resampler.h"

#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include "src/util.h"
#include "src/interpolation.h"
#include "src/Resampler/loop.h"

void LIBESPER_CDECL resampleExcitation(float* excitation, int length, int startCap, int endCap, float* output, segmentTiming timings, engineCfg config)
{
    float* idxs = (float*)malloc(length * sizeof(float));
    for (int i = 0; i < length; i++)
    {
        *(idxs + i) = i;
    }
    float* positions = (float*)malloc(timings.windowEnd * sizeof(float));
    float rate = (float)length / (float)timings.windowEnd;
    for (int i = 0; i < timings.windowEnd; i++)
    {
        *(positions + i) = i * rate;
    }
    float* realBuffer = (float*)malloc(timings.windowEnd * (config.halfTripleBatchSize + 1) * 2 * sizeof(float));
    float* imgBuffer = (float*)malloc(timings.windowEnd * (config.halfTripleBatchSize + 1) * 2 * sizeof(float));
    float* realCoeffsSrc = (float*)malloc(length * sizeof(float));
    float* imgCoeffsSrc = (float*)malloc(length * sizeof(float));
    for (int i = 0; i < config.halfTripleBatchSize + 1; i++)
    {
        for (int j = 0; j < length; j++)
        {
            *(realCoeffsSrc + j) = *(excitation + j * (config.halfTripleBatchSize + 1) + i);
            *(imgCoeffsSrc + j) = *(excitation + (j + length) * (config.halfTripleBatchSize + 1) + i);
        }
        float* realCoeffsTgt = extrap(idxs, realCoeffsSrc, positions, length, timings.windowEnd);
        float* imgCoeffsTgt = extrap(idxs, realCoeffsSrc, positions, length, timings.windowEnd);
        for (int j = 0; j < timings.windowEnd; j++)
        {
            float abs = sqrtf(powf(*(realCoeffsTgt + j), 2) + powf(*(imgCoeffsTgt + j), 2));
            float arg = (float)rand() / (float)RAND_MAX * 2.f * pi;
            *(realBuffer + j * (config.halfTripleBatchSize + 1) + i) = abs * cosf(arg);
            *(imgBuffer + j * (config.halfTripleBatchSize + 1) + i) = abs * sinf(arg);
        }
        free(realCoeffsTgt);
        free(imgCoeffsTgt);
    }
    free(idxs);
    free(positions);
    free(realCoeffsSrc);
    free(imgCoeffsSrc);
    //fade in sample if required
    if (startCap != 0)
    {
        float factor = -log2f((float)(timings.start2 - timings.start1) / (float)(timings.start3 - timings.start1));
        for (int i = timings.windowStart; i < timings.windowStart + timings.start3 - timings.start1; i++)
        {
            for (int j = 0; j < (config.halfTripleBatchSize + 1); j++)
            {
                *(realBuffer + i * (config.halfTripleBatchSize + 1) + j) *= powf((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
                *(imgBuffer + i * (config.halfTripleBatchSize + 1) + j) *= powf((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
            }
        }
    }
    //fade out sample if required
    if (endCap != 0)
    {
        float factor = -log2f((float)(timings.end3 - timings.end2) / (float)(timings.end3 - timings.end1));
        for (int i = timings.windowEnd + timings.end1 - timings.end3; i < timings.windowEnd; i++)
        {
            for (int j = 0; j < (config.halfTripleBatchSize + 1); j++)
            {
                *(realBuffer + i * (config.halfTripleBatchSize + 1) + j) *= powf(1. - ((float)(i - timings.windowEnd - timings.end1 + timings.end3 + 1) / (float)(timings.end3 - timings.end1 + 1)), factor);
                *(imgBuffer + i * (config.halfTripleBatchSize + 1) + j) *= powf(1. - ((float)(i - timings.windowEnd - timings.end1 + timings.end3 + 1) / (float)(timings.end3 - timings.end1 + 1)), factor);
            }
        }
    }
    for (int i = 0; i < (timings.windowEnd - timings.windowStart) * (config.halfTripleBatchSize + 1); i++)
    {
        *(output + i) = *(realBuffer + timings.windowStart * (config.halfTripleBatchSize + 1) + i);
        *(output + (timings.windowEnd - timings.windowStart) * (config.halfTripleBatchSize + 1) + i) = *(imgBuffer + timings.windowStart * (config.halfTripleBatchSize + 1) + i);
    }
    free(realBuffer);
    free(imgBuffer);
}

//C implementation of the ESPER specharm resampler. Respects loop spacing setting and start/end fading flags.
void LIBESPER_CDECL resampleSpecharm(float* avgSpecharm, float* specharm, int length, float* steadiness, float spacing, int startCap, int endCap, float* output, segmentTiming timings, engineCfg config)
{
    float* buffer = (float*) malloc(timings.windowEnd * config.frameSize * sizeof(float));
    //loop specharm
    loopSamplerSpecharm(specharm, length, buffer, timings.windowEnd, spacing, config);
    //scale specharm according to steadiness setting and add it to average
    #pragma omp parallel for
    for (int i = 0; i < timings.windowEnd; i++)
    {
		float multiplier = 1. - *(steadiness + i);
        for (int j = 0; j < config.halfHarmonics; j++)
        {
            *(buffer + i * config.frameSize + j) *= multiplier;
            *(buffer + i * config.frameSize + j) += *(avgSpecharm + j);
        }
        for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++)
        {
            *(buffer + i * config.frameSize + j) *= multiplier;
            *(buffer + i * config.frameSize + j) += *(avgSpecharm - config.halfHarmonics + j);//j start offset and subtraction result in addition of halfHarmonics when combined
        }
    }
    //fade in sample if required
    if (startCap != 0)
    {
        float factor = -log2f((float)(timings.start2 - timings.start1) / (float)(timings.start3 - timings.start1));
        for (int i = timings.windowStart; i < timings.windowStart + timings.start3 - timings.start1; i++)
        {
            for (int j = 0; j < config.halfHarmonics; j++)
            {
                *(buffer + i * config.frameSize + j) *= powf((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
            }
            for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++)
            {
                *(buffer + i * config.frameSize + j) *= powf((float)(i - timings.windowStart + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
            }
        }
    }
    //fade out sample if required
    if (endCap != 0)
    {
        float factor = -log2f((float)(timings.end3 - timings.end2) / (float)(timings.end3 - timings.end1));
        for (int i = timings.windowEnd + timings.end1 - timings.end3; i < timings.windowEnd; i++)
        {
            for (int j = 0; j < config.halfHarmonics; j++)
            {
                *(buffer + i * config.frameSize + j) *= powf(1. - ((float)(i - timings.windowEnd - timings.end1 + timings.end3 + 1) / (float)(timings.end3 - timings.end1 + 1)), factor);
            }
            for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++)
            {
                *(buffer + i * config.frameSize + j) *= powf(1. - ((float)(i - timings.windowEnd - timings.end1 + timings.end3 + 1) / (float)(timings.end3 - timings.end1 + 1)), factor);
            }
        }
    }
    //fill output buffer
    #pragma omp parallel for
    for (int i = 0; i < (timings.windowEnd - timings.windowStart) * config.frameSize; i++)
    {
        *(output + i) = *(buffer + timings.windowStart * config.frameSize + i);
    }
    free(buffer);
}

//C implementation of the ESPER pitch resampler. Respects loop spacing setting and start/end fading flags.
void LIBESPER_CDECL resamplePitch(int* pitchDeltas, int length, float pitch, float spacing, int startCap, int endCap, float* output, int requiredSize, segmentTiming timings)
{
    //loop pitch
    loopSamplerPitch(pitchDeltas, length, output, requiredSize, spacing);
    //load data into output buffer
    #pragma omp parallel for
    for (int i = 0; i < requiredSize; i++)
    {
        *(output + i) -= pitch;
    }
    //fade in if required
    if (startCap == 0)
    {
        float factor = -log2f((float)(timings.start2 - timings.start1) / (float)(timings.start3 - timings.start1));
        for (int i = 0; i < timings.start3 - timings.start1; i++)
        {
            *(output + i) *= powf((float)(i + 1) / (float)(timings.start3 - timings.start1 + 1), factor);
        }
    }
    //fade out if required
    if (endCap == 0)
    {
        float factor = -log2f((float)(timings.end3 - timings.end2) / (float)(timings.end3 - timings.end1));
        for (int i = requiredSize - timings.end3 + timings.end1; i < requiredSize; i++)
        {
            *(output + i) *= powf(1. - ((float)(i - (requiredSize - timings.end3 + timings.end1) + 1) / (float)(timings.end3 - timings.end1 + 1)), factor);
        }
    }
}
