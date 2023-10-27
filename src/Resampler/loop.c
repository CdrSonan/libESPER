#include "src/Resampler/loop.h"

#include <malloc.h>
#include "src/util.h"
#include "src/interpolation.h"

//loops a specharm with configurable overlap between instances.
//An appropriate blending algorithm is used for creating the transitions for the harmonics phases portion of the specharm.
void loopSamplerSpecharm(float* input, int length, float* output, int targetLength, float spacing, engineCfg config)
{
    int effSpacing = ceildiv(spacing * length,  2);
    int requiredInstances = targetLength / (length - effSpacing);
    int lastWin = targetLength - requiredInstances * (length - effSpacing);
    printf("effSpacing: %i, requiredInstances: %i, lastWin: %i\n", effSpacing, requiredInstances, lastWin);
    if (targetLength <= length)
    {
        //one instance is required to cover the entire length.
        //Just copy the data instead of looping.
        for (int i = 0; i < targetLength * config.frameSize; i++)
        {
            *(output + i) = *(input + i);
        }
    }
    else
    {
        float* buffer = (float*) malloc((length - effSpacing) * config.frameSize * sizeof(float)); //allocate buffer
        //add first window to output and fill buffer
        for (int i = 0; i < (length - effSpacing) * config.frameSize; i++)
        {
            *(output + i) = *(input + i);
            *(buffer + i) = *(input + i);
        }
        //modify start of buffer to include transition
        for (int i = 0; i < effSpacing; i++)
        {
            for (int j = 0; j < config.halfHarmonics; j++)
            {
                *(buffer + i * config.frameSize + j) *= (float)(i + 1) / (float)(effSpacing + 1);
                *(buffer + i * config.frameSize + j) += *(input + (length - effSpacing + i) * config.frameSize + j) * (1. - ((float)(i + 1) / (float)(effSpacing + 1)));
            }
            phaseInterp_inplace(buffer + i * config.frameSize + config.halfHarmonics, input + (length - effSpacing + i) * config.frameSize + config.halfHarmonics, config.halfHarmonics, (float)(i + 1) / (float)(effSpacing + 1));
            for (int j = 2 * config.halfHarmonics; j < config.frameSize; j++)
            {
                *(buffer + i * config.frameSize + j) *= (float)(i + 1) / (float)(effSpacing + 1);
                *(buffer + i * config.frameSize + j) += *(input + (length - effSpacing + i) * config.frameSize + j) * (1. - ((float)(i + 1) / (float)(effSpacing + 1)));
            }
        }
        //add mid windows from buffer to output
        for (int i = 1; i < requiredInstances; i++)
        {
            for (int j = 0; j < (length - effSpacing) * config.frameSize; j++)
            {
                *(output + i * (length - effSpacing) * config.frameSize + j) = *(buffer + j);
            }
        }
        //add final window, which has a length below the buffer size. Use buffer data if the window is still long enough to require a transition, otherwise fall back to input data
        if (lastWin >= effSpacing)
        {
            for (int i = 0; i < lastWin * config.frameSize; i++)
            {
                *(output + requiredInstances * (length - effSpacing) + i) = *(buffer + i);
            }
        }
        else
        {
            for (int i = 0; i < lastWin * config.frameSize; i++)
            {
                *(output + requiredInstances * (length - effSpacing) + i) = *(input + length - effSpacing + i);
            }
        }
        free(buffer);
    }
}

//loops pitch data with configurable overlap between instances
void loopSamplerPitch(short* input, int length, float* output, int targetLength, float spacing)
{
    int effSpacing = ceildiv(spacing * length,  2);
    int requiredInstances = targetLength / (length - effSpacing);
    int lastWin = targetLength - requiredInstances * (length - effSpacing);
    if (targetLength <= length)
    {
        //only one instance required
        for (int i = 0; i < targetLength; i++)
        {
            *(output + i) = (float)*(input + i);
        }
    }
    else
    {
        float* buffer = (float*) malloc((length - effSpacing) * sizeof(float)); //allocate buffer
        //add first window to output and fill buffer
        for (int i = 0; i < (length - effSpacing); i++)
        {
            *(output + i) = (float)*(input + i);
            *(buffer + i) = (float)*(input + i);
        }
        //modify start of buffer to include transition
        for (int i = 0; i < effSpacing; i++)
        {
            *(buffer + i) *= (float)(i + 1) / (float)(effSpacing + 1);
            *(buffer + i) += (float)*(input + length - effSpacing + i) * (1. - ((float)(i + 1) / (float)(effSpacing + 1)));
        }
        //add mid windows from buffer to output
        for (int i = 1; i < requiredInstances; i++)
        {
            for (int j = 0; j < (length - effSpacing); j++)
            {
                *(output + i * (length - effSpacing) + j) = *(buffer + j);
            }
        }
        //add final window, which has a length below the buffer size. Use buffer data if the window is still long enough to require a transition, otherwise fall back to input data
        if (lastWin > effSpacing)
        {
            for (int i = 0; i < lastWin; i++)
            {
                *(output + requiredInstances * (length - effSpacing) + i) = *(buffer + i);
            }
        }
        else
        {
            for (int i = 0; i < lastWin; i++)
            {
                *(output + requiredInstances * (length - effSpacing) + i) = (float)*(input + length - effSpacing + i);
            }
        }
        free(buffer);
    }
}
