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
#include LIBESPER_NFFT_INCLUDE_PATH

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
void mergeSpectra(cSample sample, float* lowSpectra, float* highSpectra, engineCfg config)
{
    //slope used for blending lowSpectra and highSpectra
    float* slope = (float*)malloc((config.halfTripleBatchSize + 1) * sizeof(float));
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
}

void tempSmoothSpectra(float* sourceSpectra, float* targetSpectra, cSample sample, engineCfg config)
{
    //variable for the length of the data in the time dimension, with padding on both sides
    unsigned int timeSize = sample.config.batches + 2 * sample.config.tempDepth;
    //allocate buffers
    float* workingSpectra = (float*)malloc(timeSize * (config.halfTripleBatchSize + 1) * sizeof(float));
    //copy data to buffers and add padding
    for (int i = 0; i < sample.config.tempDepth; i++)
    {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(workingSpectra + i * (config.halfTripleBatchSize + 1) + j) = *(sourceSpectra + j);
            *(targetSpectra + i * (config.halfTripleBatchSize + 1) + j) = *(sourceSpectra + j);
        }
    }
    for (int i = sample.config.tempDepth; i < sample.config.tempDepth + sample.config.batches; i++)
    {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(workingSpectra + i * (config.halfTripleBatchSize + 1) + j) = *(sourceSpectra + (i - sample.config.tempDepth) * (config.halfTripleBatchSize + 1) + j);
            *(targetSpectra + i * (config.halfTripleBatchSize + 1) + j) = *(sourceSpectra + (i - sample.config.tempDepth) * (config.halfTripleBatchSize + 1) + j);
        }
    }
    for (int i = sample.config.tempDepth + sample.config.batches; i < timeSize; i++)
    {
        for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
        {
            *(workingSpectra + i * (config.halfTripleBatchSize + 1) + j) = *(sourceSpectra + (sample.config.batches - 1) * (config.halfTripleBatchSize + 1) + j);
            *(targetSpectra + i * (config.halfTripleBatchSize + 1) + j) = *(sourceSpectra + (sample.config.batches - 1) * (config.halfTripleBatchSize + 1) + j);
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
                    }
                    else if (j - l < 0)
                    {
                        lowJ += timeSize;
                    }
                    //perform running-mean smoothing
                    *(targetSpectra + j * (config.halfTripleBatchSize + 1) + k) += *(workingSpectra + (highJ + l) * (config.halfTripleBatchSize + 1) + k) + *(workingSpectra + (lowJ - l) * (config.halfTripleBatchSize + 1) + k);
                }
                //normalize result
                *(targetSpectra + j * (config.halfTripleBatchSize + 1) + k) /= 2 * sample.config.tempWidth + 1;
            }
        }
        //take maximum of both buffers
        for (int j = 0; j < timeSize * (config.halfTripleBatchSize + 1); j++)
        {
            if (*(workingSpectra + j) > *(targetSpectra + j))
            {
                *(targetSpectra + j) = *(workingSpectra + j);
            }
            *(workingSpectra + j) = *(targetSpectra + j);
        }
    }
    free(workingSpectra);
}

void finalizeSpectra(cSample sample, float* lowSpectra, float* highSpectra, engineCfg config)
{
    mergeSpectra(sample, lowSpectra, highSpectra, config);
    free(highSpectra);
    float* tgtSpectra = (float*)malloc((sample.config.batches + 2 * sample.config.tempDepth) * (config.halfTripleBatchSize + 1) * sizeof(float));
    tempSmoothSpectra(lowSpectra, tgtSpectra, sample, config);
    free(lowSpectra);
    //load result into the appropriate portion of sample.specharm
    for (int i = 0; i < sample.config.batches; i++)
    {
        for (int j = 0; j < (config.halfTripleBatchSize + 1); j++)
        {
            if (*(tgtSpectra + (sample.config.tempDepth + i) * (config.halfTripleBatchSize + 1) + j) < 0.001)
            {
                *(sample.specharm + i * config.frameSize + 2 * config.halfHarmonics + j) = 0.001;
            }
            else
            {
                *(sample.specharm + i * config.frameSize + 2 * config.halfHarmonics + j) = *(tgtSpectra + (sample.config.tempDepth + i) * (config.halfTripleBatchSize + 1) + j);
            }
        }
    }
    free(tgtSpectra);
}

//struct for holding the output of the pitch marker calculator.
//essentially a double array with variable length.
typedef struct
{
    double* markers;
    unsigned int markerLength;
    float* wave;
    int waveLength;
    dynIntArray transitionsUp;
    dynIntArray transitionsDown;
    dynIntArray markersUp;
    dynIntArray markersDown;
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

void seedPitchMarkers(PitchMarkerStruct* markers, cSample sample, engineCfg config)
{
    //allocate dynarrays for markers
    dynIntArray_init(&markers->markersUp);
    dynIntArray_init(&markers->markersDown);
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
        if (markers->transitionsUp.length > markers->transitionsDown.length)
        {
            limit = markers->transitionsDown.length;
        }
        else
        {
            limit = markers->transitionsUp.length;
        }
        //fallback if no match is found using any possible offset
        if (offset == limit)
        {
            dynIntArray_append(&markers->markersUp, *markers->transitionsUp.content);
            dynIntArray_append(&markers->markersDown, *markers->transitionsUp.content + sample.config.pitch / 2);
            skip = 1;
            break;
        }
        //increase offset until a valid list of upTransitionCandidates for the first upwards transition is obtained
        candidateOffset = offset;
        //search for candidates within one expected wavelength from the current offset
        limit = *(markers->transitionsUp.content + offset) + sample.config.pitch;
        //limit search to the length of the waveform
        if (*(markers->transitionsDown.content + markers->transitionsDown.length - 1) < limit)
        {
            limit = *(markers->transitionsDown.content + markers->transitionsDown.length - 1);
        }
        candidateLength = findIndex(markers->transitionsUp.content, markers->transitionsUp.length, limit) - candidateOffset;//check forpossible implications of zeroTrUp.len >? original limit
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
            index = *(markers->transitionsUp.content + candidateOffset + i);
            derr = *(markers->wave + index) - *(markers->wave + index - 1);
            if (derr > maxDerr)
            {
                maxDerr = derr;
                maxIndex = index;
            }
        }
        candidateOffset = findIndex(markers->transitionsDown.content, markers->transitionsDown.length, maxIndex);
        limit = maxIndex + getLocalPitch(maxIndex, sample, config);//check if out of bounds like with previous limit
        candidateLength = findIndex(markers->transitionsDown.content, markers->transitionsDown.length, limit) - candidateOffset;
        if (candidateLength > 0)
        {
            //one or several downwards candidates found as well!
            dynIntArray_append(&markers->markersUp, maxIndex);
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
            index = *(markers->transitionsDown.content + candidateOffset + i);
            derr = *(markers->wave + index - 1) - *(markers->wave + index);
            if (derr > maxDerr)
            {
                maxDerr = derr;
                maxIndex = index;
            }
        }
        dynIntArray_append(&markers->markersDown, maxIndex);
    }
}

void findNextMarker(dynIntArray* primaryMarkers, dynIntArray* secondaryMarkers, dynIntArray* primaryTransitions, dynIntArray* secondaryTransitions, float* wave, int waveLength, cSample sample, engineCfg config)
{
    //up = primary
    //down = secondary
    int lastPrimary = *(primaryMarkers->content + primaryMarkers->length - 1);
    int lastSecondary = *(secondaryMarkers->content + secondaryMarkers->length - 1);
    int lastPitch = getLocalPitch(lastPrimary, sample, config);
    float error = -1.; //-1 denotes an "infinite" error
    float newError;
    int transition;
    //calculate next upwards marker
    //fallback "transition": if there are no actual transitions within the search range, this point will be used instead
    if (primaryMarkers->length > 1)
    {
        transition = lastPrimary + lastSecondary - *(secondaryMarkers->content + secondaryMarkers->length - 2);
        if (transition <= lastSecondary)
        {
            transition = lastSecondary + ceildiv(lastSecondary - *(secondaryMarkers->content + secondaryMarkers->length - 2), 2);
        }
    }
    else
    {
        transition = lastPrimary + lastPitch;
        if (transition <= lastSecondary)
        {
            transition = lastSecondary + ceildiv(lastPitch, 2);
        }
    }
    //ensure the transition is larger than the previous marker, even for very rapid decreases of the expected wavelength
    //set up search range
    int limit = lastPrimary + (1. - config.DIOTolerance) * lastPitch;
    if (limit < lastSecondary)
    {
        limit = lastSecondary;
    }
    //start of the window of possible transitions
    //check if limit is out of bounds (upper)
    unsigned int start = findIndex(primaryTransitions->content, primaryTransitions->length, limit);
    unsigned int end = start;
    //end of the window of possible transitions
    limit = lastPrimary + (1 + config.DIOTolerance) * lastPitch;
    if (limit >= waveLength)
    {
        limit = waveLength - 1;
    }
    //check if limit is out of bounds (lower)
    //load all transitions within the window into validTransitions
    //TODO: replace validTransitions dynIntArray with pointer offsets on zeroTransitionsUp
    while (end < primaryTransitions->length && *(primaryTransitions->content + end) <= limit)
    {
        end++;
    }
    for (int i = start; i < end; i++)
    {
        int tmpTransition = *(primaryTransitions->content + i);
        int localPitch = getLocalPitch(tmpTransition, sample, config);
        //correlate sample of one expected wavelength from current last upwards marker and candidate upwards marker
        float* sample = (float*)malloc(localPitch * sizeof(float));
        float* shiftedSample = (float*)malloc(localPitch * sizeof(float));
        if (tmpTransition + localPitch >= waveLength)
        {
            if (localPitch > lastPrimary) {
                continue;
            }
            for (int j = 0; j < localPitch; j++)
            {
                *(sample + j) = *(wave + lastPrimary - localPitch + j);
                *(shiftedSample + j) = *(wave + tmpTransition - localPitch + j);
            }
        }
        else
        {
            for (int j = 0; j < localPitch; j++)
            {
                *(sample + j) = *(wave + lastPrimary + j);
                *(shiftedSample + j) = *(wave + tmpTransition + j);
            }
        }
        //accept candidate with highest correlation
        newError = 0.;
        for (int j = 0; j < localPitch; j++)
        {
            newError += powf(*(sample + j) - *(shiftedSample + j), 2);
        }
        newError *= fabsf((float)(tmpTransition - lastPrimary - localPitch)) / (float)localPitch + config.DIOBias2;
        if (error > newError || error == -1)
        {
            transition = tmpTransition;
            error = newError;
        }
        free(sample);
        free(shiftedSample);
    }
    dynIntArray_append(primaryMarkers, transition);
}

//calculates precise pitch markers for the padded waveform of a sample.
//There is one marker for each vocal chord vibration, and they are pitch-synchronous.
//The phase angle of each marker with respect to the f0 is constant, though its exact value is arbitrary and depends on the shape of the waveform.
//This algorithm was originally based on DIO, but has since been adapted and heavily modified.
PitchMarkerStruct calculatePitchMarkers(cSample sample, float* wave, int waveLength, engineCfg config)
{
    PitchMarkerStruct markers;
    markers.wave = wave;
    markers.waveLength = waveLength;
    //get all zero transitions and load them into dynamic arrays
    dynIntArray_init(&markers.transitionsUp);
    dynIntArray_init(&markers.transitionsDown);
    for (int i = 2; i < waveLength; i++)
    {
        if ((*(wave + i - 1) < 0) && (*(wave + i) >= 0))
        {
            dynIntArray_append(&markers.transitionsUp, i);
        }
        if ((*(wave + i - 1) >= 0) && (*(wave + i) < 0))
        {
            dynIntArray_append(&markers.transitionsDown, i);
        }
    }
    //check if there are enough transitions to continue
    if (markers.transitionsUp.length <= 1 || markers.transitionsDown.length <= 1)
    {
        //not enough transitions; return two markers describing the approximate pitch
        markers.markers = (double*) malloc (2 * sizeof(double));
        *markers.markers = 0.;
        *(markers.markers + 1) = (double)sample.config.pitch;
        markers.markerLength = 2;
        dynIntArray_dealloc(&markers.transitionsUp);
        dynIntArray_dealloc(&markers.transitionsDown);
        return markers;
    }
    seedPitchMarkers(&markers, sample, config);
    //we now have an initial upTransitionMarker, followed by an initial downTransitionMarker within one expected wavelength.
    //With this, we can jump-start the algorithm.
    //loops until the entire sample is covered with markers
    while (*(markers.markersDown.content + markers.markersDown.length - 1) < markers.waveLength - (int)(*(sample.pitchDeltas + sample.config.pitchLength - 1) * config.DIOLastWinTolerance))//check for negative out-of-bounds
    {
        findNextMarker(&markers.markersUp, &markers.markersDown, &markers.transitionsUp, &markers.transitionsDown, markers.wave, markers.waveLength, sample, config);
        findNextMarker(&markers.markersDown, &markers.markersUp, &markers.transitionsDown, &markers.transitionsUp, markers.wave, markers.waveLength, sample, config);
    }
    dynIntArray_dealloc(&markers.transitionsUp);
    dynIntArray_dealloc(&markers.transitionsDown);

    //truncate final markers, if necessary
    if (*(markers.markersDown.content + markers.markersDown.length - 1) >= waveLength)
    {
        markers.markersUp.length--;
        markers.markersDown.length--;
    }
    //check if sufficient markers have been found, and use fallback if that is not the case
    if (markers.markersUp.length <= 1)
    {
        dynIntArray_dealloc(&markers.markersUp);
        dynIntArray_dealloc(&markers.markersDown);
        markers.markers = (double*) malloc (2 * sizeof(double));
        *markers.markers = 0;
        *(markers.markers + 1) = sample.config.pitch;
        markers.markerLength = 2;
        return markers;
    }
    //fill output struct with average between upwards and downwards marker for each wavelength
    markers.markerLength = markers.markersUp.length;
    markers.markers = (double*) malloc(markers.markerLength * sizeof(double));
    for (int i = 0; i < markers.markerLength; i++)
    {
        *(markers.markers + i) = (double)(*(markers.markersUp.content + i) + *(markers.markersDown.content + i)) / 2.;
        //*(markers.markers + i) = floor(*(markers.markers + i)); Last resort option should 1-sample timing mismatch between wave and re-synthesized voiced signal not be resolvable otherwise
    }
    dynIntArray_dealloc(&markers.markersUp);
    dynIntArray_dealloc(&markers.markersDown);
    return markers;
}

typedef struct
{
    float* evaluationPoints;
    int offset;
    float* offsetWindow;
    int size;
} evaluationPointsStruct;

void separateVoicedUnvoicedSingleWindow(int index, float* wave, int windowLength, evaluationPointsStruct* eval, fftw_complex* result, PitchMarkerStruct markers, cSample sample, engineCfg config)
{
    float* window = wave + index * config.batchSize;
    //get fitting segment of marker array
    int localMarkerStart = findIndex_double(markers.markers, markers.markerLength, index * config.batchSize) - 1;
    if (localMarkerStart < 0)
    {
        localMarkerStart = 0;
    }
    int localMarkerEnd = findIndex_double(markers.markers, markers.markerLength, index * config.batchSize + config.tripleBatchSize * config.filterBSMult);
    if (localMarkerEnd >= markers.markerLength)
    {
        localMarkerEnd = markers.markerLength - 1;
    }
    float* offsetWindow = wave + (int)*(markers.markers + localMarkerStart);
    int offsetWindowLength = (int)*(markers.markers + localMarkerEnd) - (int)floor(*(markers.markers + localMarkerStart)) + 1;
    int windowOffset = index * config.batchSize - (int)floor(*(markers.markers + localMarkerStart));
    int markerLength = localMarkerEnd - localMarkerStart + 1;
    float* evaluationPoints;
    //check if there are sufficient markers to perform pitch-synchronous analysis
    if (markerLength <= 1)
    {
        evaluationPoints = (float*)malloc(windowLength * sizeof(float));
        for (int j = 0; j < windowLength; j++)
        {
            *(evaluationPoints + j) = (float)j / sample.config.pitch;
        }
        offsetWindow = window;
        offsetWindowLength = windowLength;
        windowOffset = 0;
        markerLength = 2;
    }
    else if (markerLength == 2)
    {
        evaluationPoints = (float*)malloc(offsetWindowLength * sizeof(float));
        for (int j = 0; j < offsetWindowLength; j++)
        {
            *(evaluationPoints + j) = (j - windowOffset) / sample.config.pitch;
        }
        markerLength = 2;
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
        //evaluation points contain pitch-synchronous coordinate for each waveform element in the offset window
        evaluationPoints = extrap(localMarkers, markerSpace, windowSpace, markerLength, offsetWindowLength);
        free(localMarkers);
        free(markerSpace);
        free(windowSpace);
    }

    nfft_plan combinedNUFFT;
    nfft_init_1d(&combinedNUFFT, config.nHarmonics * 2, offsetWindowLength);
    for (int i = 0; i < offsetWindowLength; i++)
    {
        combinedNUFFT.x[i] = fmodf(0.5 * *(evaluationPoints + i), 1.f);
        if (combinedNUFFT.x[i] > 0.5)
        {
            combinedNUFFT.x[i] -= 1.;
        }
    }
    if (combinedNUFFT.flags & PRE_ONE_PSI)
    {
        nfft_precompute_one_psi(&combinedNUFFT);
    }
    float* hannWindowInst = hannWindow(offsetWindowLength, 3. / (float)offsetWindowLength);
    for (int i = 0; i < offsetWindowLength; i++)
    {
        (*(combinedNUFFT.f + i))[0] = *(offsetWindow + i) * *(hannWindowInst + i);
        (*(combinedNUFFT.f + i))[1] = 0.;
    }
    free(hannWindowInst);
    nfft_adjoint_1d(&combinedNUFFT);
    for (int i = 0; i < config.nHarmonics + 2; i++)
    {
        (*(result + index * (config.nHarmonics + 2) + i))[0] = (*(combinedNUFFT.f_hat + i))[0];
        (*(result + index * (config.nHarmonics + 2) + i))[1] = (*(combinedNUFFT.f_hat + i))[1];
    }
    nfft_finalize(&combinedNUFFT);
    eval->evaluationPoints = evaluationPoints;
    eval->offset = windowOffset;
    eval->offsetWindow = offsetWindow;
    eval->size = offsetWindowLength;
}

float calculatePhaseContinuity(float phaseA, float phaseB)
{
    float phaseDiff = fmin(phaseB - phaseA, phaseA - phaseB);
    return cos(phaseDiff / 2.);
}

void separateVoicedUnvoicedPostProc(fftw_complex* result, cSample sample, engineCfg config)
{
    for (int i = 0; i < sample.config.batches; i++)
    {
        for (int j = 0; j < config.halfHarmonics; j++)
            //2j+1-th component contains unvoiced part of 2j-th component
            //> 2j+1-th component calculated as mean(2j-1, 2j+1) + 2ndDerr as (2*2j - 2j-2 - 2j+2)/2
            //> 2j+1 = (2j-1 + 2j+1 - 2j-2/2 - 2j+2/2 + 2j)/2
            //store original abs in imaginary part
        {
            (*(result + i * (config.nHarmonics + 2) + 2 * j + 1))[1] = cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j + 1));
        }
        (*(result + i * (config.nHarmonics + 2) + 1))[0] =
            (
                (*(result + i * (config.nHarmonics + 2) + 1))[1] -
                cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2)) / 2. +
                cpxAbsd(*(result + i * (config.nHarmonics + 2))) / 2.
                );
        for (int j = 1; j < config.halfHarmonics - 1; j++)
        {
            (*(result + i * (config.nHarmonics + 2) + 2 * j + 1))[0] =
                (
                    (*(result + i * (config.nHarmonics + 2) + 2 * j - 1))[1] +
                    (*(result + i * (config.nHarmonics + 2) + 2 * j + 1))[1] -
                    cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j - 2)) / 2. -
                    cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j + 2)) / 2. +
                    cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j))
                    ) / 2.;
        }
        int j = config.halfHarmonics - 1;
        (*(result + i * (config.nHarmonics + 2) + 2 * j + 1))[0] =
            (
                (*(result + i * (config.nHarmonics + 2) + 2 * j - 1))[1] -
                cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j - 2)) / 2. +
                cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j)) / 2.
                );
    }
    for (int i = 0; i < config.halfHarmonics; i++)
    {
        (*(result + 2 * i + 1))[1] =
            (
                (*(result + 2 * i + 1))[0] * 2. +
                (*(result + (config.nHarmonics + 2) + 2 * i + 1))[0]
                ) / 3.;
        for (int j = 1; j < sample.config.batches - 1; j++)
        {
            (*(result + j * (config.nHarmonics + 2) + 2 * i + 1))[1] =
                (
                    (*(result + j * (config.nHarmonics + 2) + 2 * i + 1))[0] * 2. +
                    (*(result + (j - 1) * (config.nHarmonics + 2) + 2 * i + 1))[0] +
                    (*(result + (j + 1) * (config.nHarmonics + 2) + 2 * i + 1))[0]
                    ) / 4.;

        }
        (*(result + (sample.config.batches - 1) * (config.nHarmonics + 2) + 2 * i + 1))[1] =
            (
                (*(result + (sample.config.batches - 1) * (config.nHarmonics + 2) + 2 * i + 1))[0] * 2. +
                (*(result + (sample.config.batches - 2) * (config.nHarmonics + 2) + 2 * i + 1))[0]
                ) / 3.;
    }
}

void separateVoicedUnvoicedFinalize(evaluationPointsStruct* evals, fftw_complex* result, float* wave, float* unvoicedSignal, float* hannWindowInst, cSample sample, engineCfg config)
{
    for (int i = 0; i < sample.config.batches; i++)
    {
        for (int j = 0; j < config.halfHarmonics; j++)
        {
            float reducer = fmaxf((*(result + i * (config.nHarmonics + 2) + 2 * j + 1))[0], (*(result + i * (config.nHarmonics + 2) + 2 * j + 1))[1]) * sample.config.voicedThrh;
            *(sample.specharm + i * config.frameSize + config.halfHarmonics - 1 - j) = fmaxf(cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j)) - reducer, 0.);
            *(sample.specharm + i * config.frameSize + config.nHarmonics + 1 - j) = cpxArgd(*(result + i * (config.nHarmonics + 2) + 2 * j));
        }
        nfft_plan inverseNUFFT;
        nfft_init_1d(&inverseNUFFT, config.nHarmonics, config.tripleBatchSize * config.filterBSMult);
        float preIncrement = *((*(evals + i)).evaluationPoints + 1) - *((*(evals + i)).evaluationPoints);
        for (int j = 0; j < -(*(evals + i)).offset; j++)
        {
            inverseNUFFT.x[j] = fmodf((j - (*(evals + i)).offset) * preIncrement + (*(evals + i)).offset, 1.f); //added (*(evals + i)).offset) because j mod 1 = 0 and it prevents value from going negative
            if (inverseNUFFT.x[j] > 0.5)
            {
                inverseNUFFT.x[j] -= 1.;
            }
        }
        int lowerLimit;
        if ((*(evals + i)).offset < 0)
        {
            lowerLimit = -(*(evals + i)).offset;
        }
        else
        {
            lowerLimit = 0;
        }
        int upperLimit;
        if (-(*(evals + i)).offset + (*(evals + i)).size < config.tripleBatchSize * config.filterBSMult)
        {
            upperLimit = -(*(evals + i)).offset + (*(evals + i)).size;
        }
        else
        {
            upperLimit = config.tripleBatchSize * config.filterBSMult;
        }
        for (int j = lowerLimit; j < upperLimit; j++)
        {
            inverseNUFFT.x[j] = fmodf(*((*(evals + i)).evaluationPoints + (*(evals + i)).offset + j), 1.f);
            if (inverseNUFFT.x[j] > 0.5)
            {
                inverseNUFFT.x[j] -= 1.;
            }
        }
        float postIncrement = *((*(evals + i)).evaluationPoints + (*(evals + i)).size - 1) - *((*(evals + i)).evaluationPoints + (*(evals + i)).size - 2);
        for (int j = -(*(evals + i)).offset + (*(evals + i)).size; j < config.tripleBatchSize * config.filterBSMult; j++)
        {
            inverseNUFFT.x[j] = fmodf(j * postIncrement, 1.f);
            if (inverseNUFFT.x[j] > 0.5)
            {
                inverseNUFFT.x[j] -= 1.;
            }
        }
        free((*(evals + i)).evaluationPoints);
        if (inverseNUFFT.flags & PRE_ONE_PSI)
        {
            nfft_precompute_one_psi(&inverseNUFFT);
        }
        for (int j = 0; j < config.halfHarmonics - 1; j++)
        {
            inverseNUFFT.f_hat[config.nHarmonics - j - 1][0] = cos(cpxArgd(*(result + i * (config.nHarmonics + 2) + 2 * j + 2))) * cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j + 2));
            inverseNUFFT.f_hat[config.nHarmonics - j - 1][1] = sin(cpxArgd(*(result + i * (config.nHarmonics + 2) + 2 * j + 2))) * cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j + 2)) * -1.;
        }
        for (int j = 0; j < config.halfHarmonics; j++)
        {
            inverseNUFFT.f_hat[j][0] = cos(cpxArgd(*(result + i * (config.nHarmonics + 2) + 2 * j))) * cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j));
            inverseNUFFT.f_hat[j][1] = sin(cpxArgd(*(result + i * (config.nHarmonics + 2) + 2 * j))) * cpxAbsd(*(result + i * (config.nHarmonics + 2) + 2 * j));
        }
        nfft_trafo_1d(&inverseNUFFT);
        for (int j = 0; j < config.tripleBatchSize * config.filterBSMult; j++)
        {
            *(unvoicedSignal + i * config.batchSize + j) += (*(wave + i * config.batchSize + j) - inverseNUFFT.f[j][0]) * *(hannWindowInst + j);
        }
        nfft_finalize(&inverseNUFFT);
    }
}

//separates voiced and unvoiced excitation of a sample through pitch-synchronous analysis.
//requires pitch data to be included in the sample struct to work correctly.
void separateVoicedUnvoiced(cSample sample, engineCfg config)
{
    //separation calculations are only necessary if the sample is voiced
    if (sample.config.isVoiced == 0)
    {
        for (int i = 0; i < sample.config.batches; i++)
        {
            for (int j = 0; j < config.nHarmonics + 2; j++)
            {
                *(sample.specharm + i * config.frameSize + j) = 0.;
            }
        }
        stft_inpl(sample.waveform, sample.config.length, config, sample.excitation);
    }
    // extended waveform buffer aligned with batch size
    int padLength = config.halfTripleBatchSize * config.filterBSMult;
    int waveLength = sample.config.length + 2 * padLength;
    int windowLength = config.tripleBatchSize * config.filterBSMult;
    float* wave = (float*) malloc(waveLength * sizeof(float));
    float* unvoicedSignal = (float*)malloc(waveLength * sizeof(float));
    // fill input buffer, extend data with reflection padding on both sides
    for (int i = 0; i < padLength; i++)
    {
        *(wave + padLength - 1 - i) = *(sample.waveform + i);
    }
    for (int i = 0; i < sample.config.length; i++)
    {
        *(wave + padLength + i) = *(sample.waveform + i);
    }
    for (int i = 0; i < padLength; i++)
    {
        *(wave + padLength + sample.config.length + i) = *(sample.waveform + sample.config.length - 1);
    }
    for (int i = 0; i < waveLength; i++)
    {
        *(unvoicedSignal + i) = 0.f;
    }
    //Get DIO Pitch markers
    PitchMarkerStruct markers = calculatePitchMarkers(sample, wave, waveLength, config);
    fftw_complex* combinedCoeffs = (fftw_complex*)malloc(sample.config.batches * (config.nHarmonics + 2) * sizeof(fftw_complex));
    evaluationPointsStruct* evals = (evaluationPointsStruct*)malloc(sample.config.batches * sizeof(evaluationPointsStruct));
    for (int i = 0; i < sample.config.batches; i++)
    {
        separateVoicedUnvoicedSingleWindow(i, wave, windowLength, evals + i, combinedCoeffs, markers, sample, config);
    }
    free(markers.markers);
    separateVoicedUnvoicedPostProc(combinedCoeffs, sample, config);
    float* hannWindowInst = hannWindow(windowLength, 1. / 3.);
    separateVoicedUnvoicedFinalize(evals, combinedCoeffs, wave, unvoicedSignal, hannWindowInst, sample, config);
    free(wave);
    free(combinedCoeffs);
    free(evals);
    free(hannWindowInst);
    stft_inpl(unvoicedSignal + padLength, sample.config.length, config, sample.excitation);
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
}

void dampenOutliers(cSample sample, engineCfg config)
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
    free(variances);
}

void applySpectrumToExcitation(cSample sample, engineCfg config)
{
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

void finalizeSample(cSample sample, engineCfg config)
{
    averageSpectra(sample, config);
    if (sample.config.useVariance > 0)
    {
        dampenOutliers(sample, config);
    }
    applySpectrumToExcitation(sample, config);
}