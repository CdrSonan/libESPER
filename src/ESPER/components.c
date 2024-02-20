//Copyright 2023 - 2024 Johannes Klatt

//This file is part of libESPER.
//libESPER is free software: you can redistribute it and /or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
//libESPER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with Nova - Vox.If not, see <https://www.gnu.org/licenses/>.

#include "src/ESPER/components.h"

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include "src/util.h"
#include "src/fft.h"
#include "src/interpolation.h"
#include LIBESPER_FFTW_INCLUDE_PATH

//spectral smoothing/envelope calculation function based on the True Envelope Estimator algorithm.
//produces diverging oscillations in the high frequency range for typical vocal spectra.
//therefore, it is only useful for low-to-mid frequencies, but produces excellent results there.
float* lowRangeSmooth(cSample sample, float* signalsAbs, engineCfg config)
{
    //scale cutoff frequency based on window size
    int specWidth = (int)((float)config.tripleBatchSize / (float)(sample.config.specWidth + 3) / fmax(sample.config.expectedPitch / 440., 1.));
    float* spectra = (float*) malloc(sample.config.batches * (config.halfTripleBatchSize + 1) * sizeof(float));
    //define fourier-space windowing function for lowpass filter
    float* cutoffWindow = (float*) malloc((config.halfTripleBatchSize / 2 + 1) * sizeof(float));
    for (int i = 0; i < specWidth / 2; i++)
    {
        *(cutoffWindow + i) = 1.;
    }
    for (int i = specWidth / 2; i < specWidth; i++)
    {
        *(cutoffWindow + i) = 1. - (float)(i - (specWidth / 2)) / (float)(ceildiv(specWidth, 2) - 1);
    }
    for (int i = specWidth; i < config.halfTripleBatchSize / 2 + 1; i++)
    {
        *(cutoffWindow + i) = 0.;
    }
    #pragma omp parallel for
    for (int i = 0; i < sample.config.batches * (config.halfTripleBatchSize + 1); i++)
    {
        *(spectra + i) = *(signalsAbs + i);
    }
    fftwf_complex* f_spectra = (fftwf_complex*)malloc(sample.config.batches * (config.halfTripleBatchSize / 2 + 1) * sizeof(fftwf_complex));
    #pragma omp parallel for
    for (int i = 0; i < sample.config.batches; i++)
    {
        fftwf_plan plan_fwd = fftwf_plan_dft_r2c_1d(config.halfTripleBatchSize + 1, spectra + i * (config.halfTripleBatchSize + 1), f_spectra + i * (config.halfTripleBatchSize / 2 + 1), FFTW_ESTIMATE);
        fftwf_plan plan_bwd = fftwf_plan_dft_c2r_1d(config.halfTripleBatchSize + 1, f_spectra + i * (config.halfTripleBatchSize / 2 + 1), spectra + i * (config.halfTripleBatchSize + 1), FFTW_ESTIMATE);
        for (int j = 0; j < sample.config.specDepth; j++)
        {
            for (int k = 0; k < config.halfTripleBatchSize + 1; k++)
            {
                if (*(signalsAbs + i * (config.halfTripleBatchSize + 1) + k) > *(spectra + i * (config.halfTripleBatchSize + 1) + k))
                {
                    *(spectra + i * (config.halfTripleBatchSize + 1) + k) = *(signalsAbs + i * (config.halfTripleBatchSize + 1) + k);
                }
            }
            fftwf_execute(plan_fwd);
            for (int k = 0; k < config.halfTripleBatchSize / 2 + 1; k++)
            {
                (*(f_spectra + i * (config.halfTripleBatchSize / 2 + 1) + k))[0] *= *(cutoffWindow + k);
                (*(f_spectra + i * (config.halfTripleBatchSize / 2 + 1) + k))[1] *= *(cutoffWindow + k);
            }
            fftwf_execute(plan_bwd);
            for (int k = 0; k < config.halfTripleBatchSize + 1; k++)
            {
                *(spectra + i * (config.halfTripleBatchSize + 1) + k) /= config.halfTripleBatchSize + 1;
            }
        }
        fftwf_destroy_plan(plan_fwd);
        fftwf_destroy_plan(plan_bwd);
    }
    free(f_spectra);
    free(cutoffWindow);
    return(spectra);
}

//spectral smoothing/envelope calculation function very loosely based on the True Envelope Estimator algorithm.
//uses running means instead of a lowpass filter.
//less accurate than the lowRangeSmooth, but remains stable at high frequencies.
float* highRangeSmooth(cSample sample, float* signalsAbs, engineCfg config) {
    //variable for size of a spectrum with right-side padding
    unsigned int specSize = config.halfTripleBatchSize + sample.config.specDepth + 1;
    float* workingSpectra = (float*) malloc(sample.config.batches * specSize * sizeof(float));
    float* spectra = (float*) malloc(sample.config.batches * specSize * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < sample.config.batches; i++)
    {
        //copy data into buffers
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(workingSpectra + i * specSize + j) = *(signalsAbs + i * (config.halfTripleBatchSize + 1) + j);
            *(spectra + i * specSize + j) = *(signalsAbs + i * (config.halfTripleBatchSize + 1) + j);
        }
        //add padding on right side
        for (int j = config.halfTripleBatchSize + 1; j < specSize; j++)
        {
            *(workingSpectra + i * specSize + j) = *(signalsAbs + i * (config.halfTripleBatchSize + 1) + config.halfTripleBatchSize);
            *(spectra + i * specSize + j) = *(signalsAbs + i * (config.halfTripleBatchSize + 1) + config.halfTripleBatchSize);
        }
    }
    //When a running mean window contains the edge of the padded spectrum, it loops around to the other edge of the spectrum.
    //This contraption handles the required logic.
    int lowK;
    int highK;
    for (int i = 0; i < sample.config.specDepth; i++)
    {
        #pragma omp parallel for
        for (int j = 0; j < sample.config.batches; j++)
        {
            for (int k = 0; k < specSize; k++)
            {
                for (int l = 1; l < sample.config.specWidth + 1; l++)
                {
                    lowK = k;
                    highK = k;
                    if (k + l >= specSize)
                    {
                        highK -= specSize;
                    } else if (k - l < 0)
                    {
                        lowK += specSize;
                    }
                    //perform running mean on workingSpectra, load result into spectra
                    *(spectra + j * specSize + k) += *(workingSpectra + j * specSize + highK + l) + *(workingSpectra + j * specSize + lowK - l);
                }
                //normalize result
                *(spectra + j * specSize + k) /= 2 * sample.config.specWidth + 1;
            }
        }
        //load maximum of (smoothed) spectra and (non-smoothed) workingSpectra into both buffers
        for (int j = 0; j < sample.config.batches * specSize; j++)
        {
            if (*(workingSpectra + j) > *(spectra + j))
            {
                *(spectra + j) = *(workingSpectra + j);
            }
            *(workingSpectra + j) = *(spectra + j);
        }
    }
    free(workingSpectra);
    //remove padding and load the result into an output buffer
    float* output = (float*) malloc(sample.config.batches * (config.halfTripleBatchSize + 1) * sizeof(float));
    #pragma omp parallel for
    for (int i = 0; i < sample.config.batches; i++)
    {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(output + i * (config.halfTripleBatchSize + 1) + j) = *(spectra + i * specSize + j);
        }
    }
    free(spectra);
    return(output);
}

//given two arrays lowSpectra and highSpectra, which are accurate spectra of an audioSample for low and high frequencies respectively,
//this function performs blending to produce a single, accurate spectrum, performs additional temporal smoothing, and ensures the result is > 0.
void finalizeSpectra(cSample sample, float* lowSpectra, float* highSpectra, engineCfg config)
{
    //slope used for blending lowSpectra and highSpectra
    float* slope = (float*) malloc((config.halfTripleBatchSize + 1) * sizeof(float));
    for (int i = 0; i < config.spectralRolloff1; i++)
    {
        *(slope + i) = 0.;
    }
    for (int i = config.spectralRolloff1; i < config.spectralRolloff2; i++)
    {
        *(slope + i) = (float)(i - config.spectralRolloff1) / (float)(config.spectralRolloff2 - config.spectralRolloff1 - 1);
    }
    for (int i = config.spectralRolloff2; i < config.halfTripleBatchSize + 1; i++)
    {
        *(slope + i) = 1.;
    }
    #pragma omp parallel for
    for (int i = 0; i < sample.config.batches; i++)
    {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            //set all elements under 0.001 threshold to 0.001
            //Since the following smoothing operation cannot lower spectral values, only increase them, this ensures all elements of the final result are >= 0.001.
            if (*(lowSpectra + i * (config.halfTripleBatchSize + 1) + j) < 0.001)
            {
                *(lowSpectra + i * (config.halfTripleBatchSize + 1) + j) = 0.001;
            }
            if (*(highSpectra + i * (config.halfTripleBatchSize + 1) + j) < 0.001)
            {
                *(highSpectra + i * (config.halfTripleBatchSize + 1) + j) = 0.001;
            }
            //blend both spectra and store the result in lowSpectra
            *(lowSpectra + i * (config.halfTripleBatchSize + 1) + j) *= 1. - *(slope + j);
            *(lowSpectra + i * (config.halfTripleBatchSize + 1) + j) += *(slope + j) * *(highSpectra + i * (config.halfTripleBatchSize + 1) + j);
        }
    }
    free(slope);
    //variable for the length of the data in the time dimension, with padding on both sides
    unsigned int timeSize = sample.config.batches + 2 * sample.config.tempDepth;
    //allocate buffers
    float* workingSpectra = (float*) malloc(timeSize * (config.halfTripleBatchSize + 1) * sizeof(float));
    float* spectra = (float*) malloc(timeSize * (config.halfTripleBatchSize + 1) * sizeof(float));
    //copy data to buffers and add padding
    for (int i = 0; i < sample.config.tempDepth; i++)
    {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(workingSpectra + i * (config.halfTripleBatchSize + 1) + j) = *(lowSpectra + j);
            *(spectra + i * (config.halfTripleBatchSize + 1) + j) = *(lowSpectra + j);
        }
    }
    for (int i = sample.config.tempDepth; i < sample.config.tempDepth + sample.config.batches; i++)
    {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(workingSpectra + i * (config.halfTripleBatchSize + 1) + j) = *(lowSpectra + (i - sample.config.tempDepth) * (config.halfTripleBatchSize + 1) + j);
            *(spectra + i * (config.halfTripleBatchSize + 1) + j) = *(lowSpectra + (i - sample.config.tempDepth) * (config.halfTripleBatchSize + 1) + j);
        }
    }
    for (int i = sample.config.tempDepth + sample.config.batches; i < timeSize; i++)
    {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(workingSpectra + i * (config.halfTripleBatchSize + 1) + j) = *(lowSpectra + (sample.config.batches - 1) * (config.halfTripleBatchSize + 1) + j);
            *(spectra + i * (config.halfTripleBatchSize + 1) + j) = *(lowSpectra + (sample.config.batches - 1) * (config.halfTripleBatchSize + 1) + j);
        }
    }
    //same contraption for handling running mean windows crossing the edge of the buffers as in highRangeSmooth()
    int lowJ;
    int highJ;
    for (int i = 0; i < sample.config.tempDepth; i++)
    {
        for (int j = 0; j < timeSize; j++)
        {
            for (int k = 0; k < config.halfTripleBatchSize + 1; k++)
            {
                for (int l = 1; l < sample.config.tempWidth + 1; l++)
                {
                    lowJ = j;
                    highJ = j;
                    if (j + l >= timeSize)
                    {
                        highJ -= timeSize;
                    } else if (j - l < 0)
                    {
                        lowJ += timeSize;
                    }
                    //perform running-mean smoothing
                    *(spectra + j * (config.halfTripleBatchSize + 1) + k) += *(workingSpectra + (highJ + l) * (config.halfTripleBatchSize + 1) + k) + *(workingSpectra + (lowJ - l) * (config.halfTripleBatchSize + 1) + k);
                }
                //normalize result
                *(spectra + j * (config.halfTripleBatchSize + 1) + k) /= 2 * sample.config.tempWidth + 1;
            }
        }
        //take maximum of both buffers
        for (int j = 0; j < timeSize * (config.halfTripleBatchSize + 1); j++)
        {
            if (*(workingSpectra + j) > *(spectra + j))
            {
                *(spectra + j) = *(workingSpectra + j);
            }
            *(workingSpectra + j) = *(spectra + j);
        }
    }
    //load result into the appropriate portion of sample.specharm
    free(workingSpectra);
    #pragma omp parallel for
    for (int i = 0; i < sample.config.batches; i++)
    {
        for (int j = 0; j < (config.halfTripleBatchSize + 1); j++)
        {
            if (*(spectra + (sample.config.tempDepth + i) * (config.halfTripleBatchSize + 1) + j) < 0.001)
            {
                *(sample.specharm + i * config.frameSize + 2 * config.halfHarmonics + j) = 0.001;
            }
            else
            {
                *(sample.specharm + i * config.frameSize + 2 * config.halfHarmonics + j) = *(spectra + (sample.config.tempDepth + i) * (config.halfTripleBatchSize + 1) + j);
            }
        }
    }
    free(spectra);
}

//struct for holding the output of the pitch marker calculator.
//essentially a double array with variable length.
typedef struct
{
    double* markers;
    unsigned int markerLength;
}
PitchMarkerStruct;

//utility function for fetching the approximate pitch of a sample at a location given in data points from the start of the sample.
//requires the sample to have existing pitch data.
int getLocalPitch(int position, cSample sample, engineCfg config)
{
    //remove implicit padding
    int effectivePos = position - config.halfTripleBatchSize * (int)config.filterBSMult;
    //limit result to sample bounds and divide by batchSize to get the correct batch index.
    //this is done in this somewhat awkward order to prevent issues with unsigned integer division.
    if (effectivePos <= 0)
    {
        effectivePos = 0;
    }
    else
    {
        effectivePos /= config.batchSize;
    }
    if (effectivePos >= sample.config.pitchLength )
    {
        effectivePos = sample.config.pitchLength - 1;
    }
    return *(sample.pitchDeltas + effectivePos);
}

//calculates precise pitch markers for the padded waveform of a sample.
//There is one marker for each vocal chord vibration, and they are pitch-synchronous.
//The phase angle of each marker with respect to the f0 is constant, though its exact value is arbitrary and depends on the shape of the waveform.
//This algorithm was originally based on DIO, but has since been adapted and heavily modified.
PitchMarkerStruct calculatePitchMarkers(cSample sample, float* wave, int waveLength, engineCfg config)
{
    PitchMarkerStruct output;
    //get all zero transitions and load them into dynamic arrays
    dynIntArray zeroTransitionsUp;
    dynIntArray_init(&zeroTransitionsUp);
    dynIntArray zeroTransitionsDown;
    dynIntArray_init(&zeroTransitionsDown);
    for (int i = 2; i < waveLength; i++)
    {
        if ((*(wave + i - 1) < 0) && (*(wave + i) >= 0))
        {
            dynIntArray_append(&zeroTransitionsUp, i);
        }
        if ((*(wave + i - 1) >= 0) && (*(wave + i) < 0))
        {
            dynIntArray_append(&zeroTransitionsDown, i);
        }
    }
    //check if there are enough transitions to continue
    if (zeroTransitionsUp.length <= 1 || zeroTransitionsDown.length <= 1)
    {
        //not enough transitions; return two markers describing the approximate pitch
        output.markers = (double*) malloc (2 * sizeof(double));
        *output.markers = 0.;
        *(output.markers + 1) = (double)sample.config.pitch;
        output.markerLength = 2;
        return output;
    }
    //allocate dynarrays for markers
    dynIntArray upTransitionMarkers;
    dynIntArray_init(&upTransitionMarkers);
    dynIntArray downTransitionMarkers;
    dynIntArray_init(&downTransitionMarkers);
    //determine first relevant transition
    unsigned int offset = 0;
    short skip = 0;
    unsigned int candidateOffset;
    unsigned short candidateLength;
    int limit; //reused for various limits in the following code
    float maxDerr = 0.;
    float derr;
    int maxIndex;
    int index;
    while (1)
    {
        //the length of the shorter zeroTransition array is a hard upper limit for the offset, since it is impossible to find further transition beyond it
        if (zeroTransitionsUp.length > zeroTransitionsDown.length)
        {
            limit = zeroTransitionsDown.length;
        }
        else
        {
            limit = zeroTransitionsUp.length;
        }
        //fallback if no match is found using any possible offset
        if (offset == limit)
        {
            dynIntArray_append(&upTransitionMarkers, *zeroTransitionsUp.content);
            dynIntArray_append(&downTransitionMarkers, *zeroTransitionsUp.content + sample.config.pitch / 2);
            skip = 1;
            break;
        }
        //increase offset until a valid list of upTransitionCandidates for the first upwards transition is obtained
        candidateOffset = offset;
        //search for candidates within one expected wavelength from the current offset
        limit = *(zeroTransitionsUp.content + offset) + getLocalPitch(*(zeroTransitionsDown.content + zeroTransitionsDown.length - 1), sample, config);
        //limit search to the length of the waveform
        if (*(zeroTransitionsDown.content + zeroTransitionsDown.length - 1) < limit)
        {
            limit = *(zeroTransitionsDown.content + zeroTransitionsDown.length - 1);
        }
        candidateLength = findIndex(zeroTransitionsUp.content, zeroTransitionsUp.length, limit) - candidateOffset;//check forpossible implications of zeroTrUp.len >? original limit
        if (candidateLength == 0)
        {
            //no candidates found; increase offset and try again
            offset++;
            continue;
        }
        //one or several candidates found!
        //select candidate with highest derivative
        for (int i = 0; i < candidateLength; i++)
        {
            index = *(zeroTransitionsUp.content + candidateOffset + i);
            derr = *(wave + index) - *(wave + index - 1);
            if (derr > maxDerr)
            {
                maxDerr = derr;
                maxIndex = index;
            }
        }
        candidateOffset = findIndex(zeroTransitionsDown.content, zeroTransitionsDown.length, maxIndex);
        limit = maxIndex + getLocalPitch(maxIndex, sample, config);//check if out of bounds like with previous limit
        candidateLength = findIndex(zeroTransitionsDown.content, zeroTransitionsDown.length, limit) - candidateOffset;
        if (candidateLength > 0)
        {
            //one or several downwards candidates found as well!
            dynIntArray_append(&upTransitionMarkers, maxIndex);
            break;
        }
        //no downwards candidates found; increase offset and try again
        offset++;
    }
    if (skip == 0) //skipped if the fallback function was invoked and there are not actually any downwards candidates
    {
        //select the downward transition candidate with the lowest derivative
        maxDerr = 0.;
        for (int i = 0; i < candidateLength; i++)
        {
            index = *(zeroTransitionsDown.content + candidateOffset + i);
            derr = *(wave + index - 1) - *(wave + index);
            if (derr > maxDerr)
            {
                maxDerr = derr;
                maxIndex = index;
            }
        }
        dynIntArray_append(&downTransitionMarkers, maxIndex);
    }
    //we now have an initial upTransitionMarker, followed by an initial downTransitionMarker within one expected wavelength.
    //With this, we can jump-start the algorithm.
    int lastPitch;
    int lastDown;
    int lastUp;
    float error;
    float newError;
    int transition;
    int start;
    int tmpTransition;
    int localPitch;
    dynIntArray validTransitions;
    //loops until the entire sample is covered with markers
    while (*(downTransitionMarkers.content + downTransitionMarkers.length - 1) < waveLength - (int)(*(sample.pitchDeltas + sample.config.pitchLength - 1) * config.DIOLastWinTolerance))//check for negative out-of-bounds
    {
        //convenience variables
        lastUp = *(upTransitionMarkers.content + upTransitionMarkers.length - 1);
        lastDown = *(downTransitionMarkers.content + downTransitionMarkers.length - 1);
        lastPitch = getLocalPitch(lastUp, sample, config);
        error = -1; //-1 denotes an "infinite" error
        //new, reusable array for transition candidates
        dynIntArray_init(&validTransitions);
        //calculate next upwards marker
        //fallback "transition": if there are no actual transitions within the search range, this point will be used instead
        if (upTransitionMarkers.length > 1)
        {
            transition = lastUp + lastDown - *(downTransitionMarkers.content + downTransitionMarkers.length - 2);
            if (transition <= lastDown)
            {
                transition = lastDown + ceildiv(lastDown - *(downTransitionMarkers.content + downTransitionMarkers.length - 2), 2);
            }
        }
        else
        {
            transition = lastUp + lastPitch;
            if (transition <= lastDown)
            {
                transition = lastDown + ceildiv(lastPitch, 2);
            }
        }
        //ensure the transition is larger than the previous marker, even for very rapid decreases of the expected wavelength
        //set up search range
        limit = lastUp + (1. - config.DIOTolerance) * lastPitch;
        if (limit < lastDown)
        {
            limit = lastDown;
        }
        //start of the window of possible transitions
        //check if limit is out of bounds (upper)
        start = findIndex(zeroTransitionsUp.content, zeroTransitionsUp.length, limit);
        //end of the window of possible transitions
        limit = lastUp + (1 + config.DIOTolerance) * lastPitch;
        if (limit >= waveLength)
        {
            limit = waveLength - 1;
        }
        //check if limit is out of bounds (lower)
        //load all transitions within the window into validTransitions
        //TODO: replace validTransitions dynIntArray with pointer offsets on zeroTransitionsUp
        while (start < zeroTransitionsUp.length && *(zeroTransitionsUp.content + start) <= limit)
        {
            dynIntArray_append(&validTransitions, *(zeroTransitionsUp.content + start));
            start++;
        }
        for (int i = 0; i < validTransitions.length; i++)
        {
            tmpTransition = *(validTransitions.content + i);
            localPitch = getLocalPitch(tmpTransition, sample, config);
            //correlate sample of one expected wavelength from current last upwards marker and candidate upwards marker
            float* sample = (float*) malloc(localPitch * sizeof(float));
            float* shiftedSample = (float*) malloc(localPitch * sizeof(float));
            if (tmpTransition + localPitch >= waveLength)
            {
                if (localPitch > lastUp) {
                    continue;
                }
                for (int j = 0; j < localPitch; j++)
                {
                    *(sample + j) = *(wave + lastUp - localPitch + j);
                    *(shiftedSample + j) = *(wave + tmpTransition - localPitch + j);
                }
            }
            else
            {
                for (int j = 0; j < localPitch; j++)
                {
                    *(sample + j) = *(wave + lastUp + j);
                    *(shiftedSample + j) = *(wave + tmpTransition + j);
                }
            }
            //accept candidate with highest correlation
            newError = 0.;
            for (int j = 0; j < localPitch; j++)
            {
                newError += powf(*(sample + j) - *(shiftedSample + j), 2);
            }
            newError *= fabsf((float)(tmpTransition - lastUp - localPitch)) / (float)localPitch + config.DIOBias2;
            if (error > newError || error == -1)
            {
                transition = tmpTransition;
                error = newError;
            }
        }
        dynIntArray_append(&upTransitionMarkers, transition);

        //do the same again for the next downwards transition
        lastUp = transition;
        error = -1;
        dynIntArray_dealloc(&validTransitions);
        dynIntArray_init(&validTransitions);
        transition = lastUp + lastDown - *(upTransitionMarkers.content + upTransitionMarkers.length - 2);
        if (transition < lastUp)
        {
            transition = lastUp + ceildiv(lastUp - *(upTransitionMarkers.content + upTransitionMarkers.length - 2), 2);
        }
        //check if transition is out of wave bounds (upper)
        limit = lastDown + (1. - config.DIOTolerance) * lastPitch;
        if (limit < lastUp)
        {
            limit = lastUp;
        }
        start = findIndex(zeroTransitionsDown.content, zeroTransitionsDown.length, limit);
        limit = lastDown + (1 + config.DIOTolerance) * lastPitch;
        if (limit > waveLength)
        {
            limit = waveLength;
        }
        //check if limit is out of bounds (lower)
        while (start < zeroTransitionsDown.length && *(zeroTransitionsDown.content + start) <= limit)
        {
            dynIntArray_append(&validTransitions, *(zeroTransitionsDown.content + start));
            start++;
        }
        for (int i = 0; i < validTransitions.length; i++)
        {
            tmpTransition = *(validTransitions.content + i);
            localPitch = getLocalPitch(tmpTransition, sample, config);
            float* sample = (float*) malloc(localPitch * sizeof(float));
            float* shiftedSample = (float*) malloc(localPitch * sizeof(float));
            if (tmpTransition + localPitch >= waveLength)
            {
                if (localPitch > lastDown)
                {
                    continue;
                }
                for (int j = 0; j < localPitch; j++)
                {
                    *(sample + j) = *(wave + lastDown - localPitch + j);
                    *(shiftedSample + j) = *(wave + tmpTransition - localPitch + j);
                }
            }
            else
            {
                for (int j = 0; j < localPitch; j++)
                {
                    *(sample + j) = *(wave + lastDown + j);
                    *(shiftedSample + j) = *(wave + tmpTransition + j);
                }
            }
            newError = 0.;
            for (int j = 0; j < localPitch; j++)
            {
                newError += powf(*(sample + j) - *(shiftedSample + j), 2);
            }
            newError *= fabsf((float)(tmpTransition - lastDown - localPitch)) / (float)localPitch + config.DIOBias2;
            if (error > newError || error == -1)
            {
                transition = tmpTransition;
                error = newError;
            }
        }
        dynIntArray_append(&downTransitionMarkers, transition);
        dynIntArray_dealloc(&validTransitions);
    }
    dynIntArray_dealloc(&zeroTransitionsUp);
    dynIntArray_dealloc(&zeroTransitionsDown);

    //truncate final markers, if necessary
    if (*(downTransitionMarkers.content + downTransitionMarkers.length - 1) >= waveLength)
    {
        upTransitionMarkers.length--;
        downTransitionMarkers.length--;
    }
    //check if sufficient markers have been found, and use fallback if that is not the case
    if (upTransitionMarkers.length <= 1)
    {
        output.markers = (double*) malloc (2 * sizeof(double));
        *output.markers = 0;
        *(output.markers + 1) = sample.config.pitch;
        output.markerLength = 2;
        return output;
    }
    //fill output struct with average between upwards and downwards marker for each wavelength
    output.markerLength = upTransitionMarkers.length;
    output.markers = (double*) malloc(output.markerLength * sizeof(double));
    for (int i = 0; i < output.markerLength; i++)
    {
        *(output.markers + i) = (double)(*(upTransitionMarkers.content + i) + *(downTransitionMarkers.content + i)) / 2.;
    }
    dynIntArray_dealloc(&upTransitionMarkers);
    dynIntArray_dealloc(&downTransitionMarkers);
    return output;
}

float calculatePhaseContinuity(float phaseA, float phaseB)
{
    float phaseDiff = fmin(phaseB - phaseA, phaseA - phaseB);
    return cos(phaseDiff / 2.);
}

//separates voiced and unvoiced excitation of a sample through pitch-synchronous analysis.
//Implicitely requires pitch data to be included in the sample to work correctly.
void separateVoicedUnvoiced(cSample sample, engineCfg config)
{
    // extended waveform buffer aligned with batch size
    int padLength = config.halfTripleBatchSize * config.filterBSMult;
    int waveLength = sample.config.length + 2 * padLength;
    int windowLength = config.tripleBatchSize * config.filterBSMult;
    float* wave = (float*) malloc(waveLength * sizeof(float));
    float* unvoicedSignal = (float*)malloc(waveLength * sizeof(float));
    // fill input buffer, extend data with reflection padding on both sides
    for (int i = 0; i < padLength; i++)
    {
        *(wave + i) = *(sample.waveform + padLength - i - 1);
    }
    #pragma omp parallel for
    for (int i = 0; i < sample.config.length; i++)
    {
        *(wave + padLength + i) = *(sample.waveform + i);
    }
    for (int i = 0; i < padLength; i++)
    {
        *(wave + padLength + sample.config.length + i) = *(sample.waveform + sample.config.length - 1 - i);
    }
    for (int i = 0; i < waveLength; i++)
    {
        *(unvoicedSignal + i) = 0.f;
    }
    //further variable definitions for later use
    float* hannWindow = (float*)malloc(windowLength * sizeof(float));
    for (int i = 0; i < windowLength; i++) {
        *(hannWindow + i) = pow(sin((pi / windowLength) * i), 2.) * 2. / (3. * config.filterBSMult);
    }
    int batches = sample.config.batches;
    //Get DIO Pitch markers
    PitchMarkerStruct markers = calculatePitchMarkers(sample, wave, waveLength, config);
    float* phases = (float*)malloc(config.halfHarmonics * sizeof(float));
    for (int i = 0; i < config.halfHarmonics; i++)
    {
        *(phases + i) = 1.;
    }
    float* continuity = (float*)malloc(config.halfHarmonics * sizeof(float));
    for (int i = 0; i < config.halfHarmonics; i++)
    {
        *(continuity + i) = 1.;
    }
    //Loop over each window
    for (int i = 0; i < batches; i++)
    {
        //separation calculations are only necessary if the sample is voiced
        if (sample.config.isVoiced == 0)
        {
            for (int j = 0; j < config.nHarmonics + 2; j++)
            {
                *(sample.specharm + i * config.frameSize + j) = 0.;
            }
            continue;
        }
        float* window = wave + i * config.batchSize;
        //get fitting segment of marker array
        int localMarkerStart = findIndex_double(markers.markers, markers.markerLength, i * config.batchSize) - 1;
        if (localMarkerStart < 0)
        {
            localMarkerStart = 0;
        }
        int localMarkerEnd = findIndex_double(markers.markers, markers.markerLength, i * config.batchSize + config.tripleBatchSize * config.filterBSMult);
        if (localMarkerEnd >= markers.markerLength)
        {
            localMarkerEnd = markers.markerLength - 1;
        }
        float* offsetWindow = wave + (int)*(markers.markers + localMarkerStart);
        int offsetWindowLength = (int)*(markers.markers + localMarkerEnd) - (int)ceil(*(markers.markers + localMarkerStart)) + 1;
        int windowOffset = i * config.batchSize - (int)ceil(*(markers.markers + localMarkerStart));
        int markerLength = localMarkerEnd - localMarkerStart + 1;
        fftwf_complex* harmFunction = (fftwf_complex*)malloc(config.halfHarmonics * sizeof(fftwf_complex));
        float* unvoicedSignalPart = (float*)malloc(windowLength * sizeof(float));
        float* evaluationPoints;
        //check if there are sufficient markers to perform pitch-synchronous analysis
        if (markerLength <= 1)
        {
            evaluationPoints = (float*)malloc(offsetWindowLength * sizeof(float));
            for (int j = 0; j < offsetWindowLength; j++)
            {
                *(evaluationPoints + j) = (j - windowOffset) / sample.config.pitch;
            }
            markerLength = 1;
        }
        else
        {
            //setup scales for interpolation to pitch-synchronous space
            float* localMarkers = (float*)malloc(markerLength * sizeof(float));
            float* markerSpace = (float*)malloc(markerLength * sizeof(float));
            for (int j = 0; j < markerLength; j++)
            {
                *(localMarkers + j) = *(markers.markers + localMarkerStart + j) - (int)ceil(*(markers.markers + localMarkerStart));
                *(markerSpace + j) = j;
            }
            float* windowSpace = (float*)malloc(offsetWindowLength * sizeof(float));
            for (int j = 0; j < offsetWindowLength; j++)
            {
                *(windowSpace + j) = j;
            }
            //perform interpolation and get pitch-synchronous version of waveform
            evaluationPoints = extrap(localMarkers, markerSpace, windowSpace, markerLength, offsetWindowLength);
            free(localMarkers);
            free(markerSpace);
            free(windowSpace);
        }
        float multiplier;
        for (int j = 0; j < config.halfHarmonics; j++)
        {
            (*(harmFunction + j))[0] = 0.;
            (*(harmFunction + j))[1] = 0.;
            for (int k = 0; k < offsetWindowLength; k++)
            {
                if (k == 0)
                {
                    multiplier = (*(evaluationPoints + 1) - *(evaluationPoints)) / (markerLength - 1);
                    if (ceil(*(markers.markers + localMarkerStart)) != *(markers.markers + localMarkerStart))
                    {
                        multiplier += *(evaluationPoints) / (markerLength - 1) * 2.;
                    }
                }
                else if (k == offsetWindowLength - 1)
                {
                    multiplier = (*(evaluationPoints + offsetWindowLength - 1) - *(evaluationPoints + offsetWindowLength - 2)) / (markerLength - 1);
                    if (floor(*(markers.markers + localMarkerEnd)) != *(markers.markers + localMarkerEnd))
                    {
                        multiplier += (markerLength - 1 - *(evaluationPoints + offsetWindowLength - 1)) / (markerLength - 1) * 2.;
                    }
                }
                else
                {
                    multiplier = (*(evaluationPoints + k + 1) - *(evaluationPoints + k - 1)) / (markerLength - 1);
                }
                (*(harmFunction + j))[0] += *(offsetWindow + k) * cos(-2. * pi * j * *(evaluationPoints + k)) * multiplier;

                (*(harmFunction + j))[1] += *(offsetWindow + k) * sin(-2. * pi * j * *(evaluationPoints + k)) * multiplier;
            }
            float newContinuity = calculatePhaseContinuity(*(phases + j), cpxArgf(*(harmFunction + j)));
            *(sample.specharm + i * config.frameSize + j) = cpxAbsf(*(harmFunction + j)) * newContinuity;
            if (i > 0)
            {
                *(sample.specharm + (i - 1) * config.frameSize + j) *= newContinuity;
            }
            *(sample.specharm + i * config.frameSize + config.halfHarmonics + j) = cpxArgf(*(harmFunction + j));
            *(phases + j) = cpxArgf(*(harmFunction + j));
        }
        for (int j = 0; j < windowLength; j++)
        {
            *(unvoicedSignalPart + j) = 0.;
            for (int k = 0; k < config.halfHarmonics; k++)
            {
                if (windowOffset + j < 0)
                {
                    float extrapolatedPoint = *evaluationPoints + (windowOffset + j) * (*(evaluationPoints + 1) - *evaluationPoints);
                    *(unvoicedSignalPart + j) += cpxAbsf(*(harmFunction + k)) * cos(cpxArgf(*(harmFunction + k)) + 2. * pi * k * extrapolatedPoint);
                }
                else if (windowOffset + j >= offsetWindowLength)
                {
                    float extrapolatedPoint = *(evaluationPoints + offsetWindowLength - 1) + (windowOffset + j - offsetWindowLength + 1) * (*(evaluationPoints + offsetWindowLength - 1) - *(evaluationPoints + offsetWindowLength - 2));
                    *(unvoicedSignalPart + j) += cpxAbsf(*(harmFunction + k)) * cos(cpxArgf(*(harmFunction + k)) + 2. * pi * k * extrapolatedPoint);
                }
                else
                {
                    *(unvoicedSignalPart + j) += cpxAbsf(*(harmFunction + k)) * cos(cpxArgf(*(harmFunction + k)) + 2. * pi * k * *(evaluationPoints + windowOffset + j));
                }
            }
            *(unvoicedSignalPart + j) = *(unvoicedSignalPart + j)  - *(window + j);
        }
        free(harmFunction);
        for (int j = 0; j < windowLength; j++)
        {
            *(unvoicedSignal + i * config.batchSize + j) += *(hannWindow + j) * *(unvoicedSignalPart + j);
        }
        free(unvoicedSignalPart);
    }
    free(phases);
    if (sample.config.isVoiced == 0)
    {
        stft_inpl(wave, sample.config.length, config, sample.excitation);
    }
    else
    {
        stft_inpl(unvoicedSignal, sample.config.length, config, sample.excitation);
    }
    free(unvoicedSignal);
}

//averages all harmonic amplitudes and spectra, stores the result in the avgSpecharm field of the sample, and overwrites the specharms with their difference from the average.
//avgSpecharm is shorter than a specharm, since the harmonics phases are not stored in it.
//Also dampens outlier points if the config of the sample calls for it.
//final step of the ESPER audio analysis pipeline.
void averageSpectra(cSample sample, engineCfg config)
{
    //average spectra
    for (int i = 0; i < config.halfHarmonics + config.halfTripleBatchSize + 1; i++)
    {
        *(sample.avgSpecharm + i) = 0.;
    }
    for (int i = 0; i < sample.config.batches; i++)
    {
        for (int j = 0; j < config.halfHarmonics; j++) {
            *(sample.avgSpecharm + j) += *(sample.specharm + i * config.frameSize + j);
        }
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(sample.avgSpecharm + config.halfHarmonics + j) += *(sample.specharm + i * config.frameSize + config.nHarmonics + 2 + j);
        }
    }
    for (int i = 0; i < (config.halfHarmonics + config.halfTripleBatchSize + 1); i++)
    {
        *(sample.avgSpecharm + i) /= sample.config.batches;
    }
    #pragma omp parallel for
    for (int i = 0; i < sample.config.batches; i++)
    {
        for (int j = 0; j < config.halfHarmonics; j++)
        {
            *(sample.specharm + i * config.frameSize + j) -= *(sample.avgSpecharm + j);
        }
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(sample.specharm + i * config.frameSize + config.nHarmonics + 2 + j) -= *(sample.avgSpecharm + config.halfHarmonics + j);
        }
    }
    //dampen outliers if required
    if (0)//(sample.config.useVariance > 0)
    {
        float variance = 0.;
        float* variances = (float*)malloc(sample.config.batches * sizeof(float));
        for (int i = 0; i < sample.config.batches; i++)
        {
            *(variances + i) = 0.;
        }
        for (int i = 0; i < sample.config.batches; i++) {
            for (int j = 0; j < config.halfHarmonics; j++) {
                *(variances + i) += pow(*(sample.specharm + i * config.frameSize + j), 2);
            }
            for (int j = 0; j < config.halfTripleBatchSize + 1; j++) {
                *(variances + i) += pow(*(sample.specharm + i * config.frameSize + config.nHarmonics + 2 + j), 2);
            }
        }
        for (int i = 0; i < sample.config.batches; i++) {
            *(variances + i) = sqrtf(*(variances + i));
            variance += *(variances + i);
        }
        variance /= sample.config.batches;
        for (int i = 0; i < sample.config.batches; i++) {
            *(variances + i) = *(variances + i) / variance - 1;
        }
        for (int i = 0; i < sample.config.batches; i++) {
            if (*(variances + i) > 1) {
                for (int j = 0; j < config.halfHarmonics; j++) {
                    *(sample.specharm + i * config.frameSize + j) /= *(variances + i);
                }
                for (int j = config.nHarmonics + 2; j < config.halfTripleBatchSize + 1; j++) {
                    *(sample.specharm + i * config.frameSize + j) /= *(variances + i);
                }
            }
        }
    }
    for (int i = 0; i < sample.config.batches; i++)
    {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            //divide excitation by the spectrum
            *(sample.excitation + i * (config.halfTripleBatchSize + 1) + j) /= *(sample.specharm + i * config.frameSize + config.nHarmonics + 2 + j) + *(sample.avgSpecharm + config.halfHarmonics + j);
            *(sample.excitation + (i + sample.config.batches) * (config.halfTripleBatchSize + 1) + j) /= *(sample.specharm + i * config.frameSize + config.nHarmonics + 2 + j) + *(sample.avgSpecharm + config.halfHarmonics + j);
            *(sample.excitation + i * (config.halfTripleBatchSize + 1) + j) *= 4. / config.tripleBatchSize;
            *(sample.excitation + (i + sample.config.batches) * (config.halfTripleBatchSize + 1) + j) *= 4. / config.tripleBatchSize;
        }
    }
}
