#include "src/Resampler/resampler.h"

#include <malloc.h>
#include <math.h>
#include "src/util.h"
#include "src/Resampler/loop.h"

//C implementation of the ESPER specharm resampler. Respects loop spacing setting and start/end fading flags.
__declspec(dllexport) void __cdecl resampleSpecharm(float* avgSpecharm, float* specharm, int length, float* steadiness, float spacing, int startCap, int endCap, float* output, segmentTiming timings, engineCfg config)
{
    float* buffer = (float*) malloc(timings.windowEnd * config.frameSize * sizeof(float));
    //loop specharm
    loopSamplerSpecharm(specharm, length, buffer, timings.windowEnd, spacing, config);
    //scale specharm according to steadiness setting and add it to average
    #pragma omp parallel for
    for (int i = 0; i < timings.windowEnd; i++)
    {
        for (int j = 0; j < config.halfHarmonics; j++)
        {
            *(buffer + i * config.frameSize + j) *= powf(1. - *(steadiness + i), 2.);
            *(buffer + i * config.frameSize + j) += *(avgSpecharm + j);
        }
        for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++)
        {
            *(buffer + i * config.frameSize + j) *= powf(1. - *(steadiness + i), 2.);
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
__declspec(dllexport) void __cdecl resamplePitch(short* pitchDeltas, int length, float pitch, float spacing, int startCap, int endCap, float* output, int requiredSize, segmentTiming timings)
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
