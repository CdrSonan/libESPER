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
    //legacy method: int specWidth = (int)((float)config.tripleBatchSize / (float)(sample.config.specWidth + 3) / fmax(sample.config.expectedPitch / 440., 1.));
    //int specWidth = (int)((float)config.sampleRate / 6. / (float)sample.config.expectedPitch);
    int specWidth = (float)sample.config.pitch / 6.;
    float* spectra = (float*) malloc(sample.config.batches * (config.halfTripleBatchSize + 1) * sizeof(float));
    //define fourier-space windowing function for lowpass filter
    float* cutoffWindow = (float*) malloc((config.halfTripleBatchSize / 2 + 1) * sizeof(float));
    for (int i = 0; i < specWidth / 2; i++)
    {
        *(cutoffWindow + i) = 1.;
    }
    for (int i = specWidth / 2; i < specWidth; i++)
    {
        *(cutoffWindow + i) = 1.;// -(float)(i - (specWidth / 2)) / (float)(ceildiv(specWidth, 2) - 1);
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

float* getEvaluationPoints(int start, int end, float* wave, cSample sample, engineCfg config)
{
    float* localMarkers = (float*)malloc((end - start) * sizeof(float));
    float* markerSpace = (float*)malloc((end - start) * sizeof(float));
    for (int j = 0; j < end - start; j++)
    {
        *(localMarkers + j) = *(sample.pitchMarkers + start + j) - (int)ceil(*(sample.pitchMarkers + start));
        *(markerSpace + j) = j;
    }
	int windowLength = *(sample.pitchMarkers + end) - *(sample.pitchMarkers + start);
    float* windowSpace = (float*)malloc(windowLength * sizeof(float));
    for (int j = 0; j < windowLength; j++)
    {
        *(windowSpace + j) = j;
    }
    //evaluation points contain pitch-synchronous coordinate for each waveform element in the offset window
    float* evaluationPoints = extrap(localMarkers, markerSpace, windowSpace, end - start, windowLength);
    free(localMarkers);
    free(markerSpace);
    free(windowSpace);
	return evaluationPoints;
}

void separateVoicedUnvoicedSingleWindow(int index, float* evaluationPoints, fftw_complex* result, cSample sample, engineCfg config)
{
	int windowStart = *(sample.pitchMarkers + index);
    int windowLength = *(sample.pitchMarkers + index + 1) - windowStart;
    float* windowPtr = sample.waveform + windowStart;
    
    nfft_plan combinedNUFFT;
    nfft_init_1d(&combinedNUFFT, config.nHarmonics, windowLength);
    for (int i = 0; i < windowLength; i++)
    {
        combinedNUFFT.x[i] = *(evaluationPoints + windowStart + i);
		if (combinedNUFFT.x[i] > 0.5)
		{
			combinedNUFFT.x[i] -= 1.;
		}
    }
    if (combinedNUFFT.flags & PRE_ONE_PSI)
    {
        nfft_precompute_one_psi(&combinedNUFFT);
    }
    for (int i = 0; i < windowLength; i++)
    {
        (*(combinedNUFFT.f + i))[0] = *(windowPtr + i) / (float)(windowLength);
        (*(combinedNUFFT.f + i))[1] = 0.;
    }
    nfft_adjoint_1d(&combinedNUFFT);
    for (int i = 0; i < config.halfHarmonics; i++)
    {
        (*(result + index * config.halfHarmonics + i))[0] = (*(combinedNUFFT.f_hat + i))[0];
        (*(result + index * config.halfHarmonics + i))[1] = (*(combinedNUFFT.f_hat + i))[1];
    }
    nfft_finalize(&combinedNUFFT);
}

float calculatePhaseContinuity(float phaseA, float phaseB)
{
    float phaseDiff = fmin(phaseB - phaseA, phaseA - phaseB);
    return cos(phaseDiff / 2.);
}


int compareFloats(const void* a, const void* b)
{
    float diff = (*(const float*)a - *(const float*)b);
    if (diff < 0) return -1;
    if (diff > 0) return 1;
    return 0;
}

float* gaussWindow(int length, float sigma)
{
	float* window = (float*)malloc(length * sizeof(float));
	for (int i = 0; i < length; i++)
	{
		*(window + i) = exp(-0.5 * pow((float)(i - length / 2) / sigma, 2));
	}
	//normalize window with respect to discretization error
	float sum = 0.;
	for (int i = 0; i < length; i++)
	{
		sum += *(window + i);
	}
	for (int i = 0; i < length; i++)
	{
		*(window + i) /= sum;
	}
	return window;
}

void separateVoicedUnvoicedPostProc(fftw_complex* dftCoeffs, cSample sample, engineCfg config)
{
	int effectiveLength = sample.config.markerLength - 1;
	int kernelSize = 10;
	float sigmaBase = 5;
	fftw_complex* smoothedDftCoeffs = (fftw_complex*)malloc(effectiveLength * config.halfHarmonics * sizeof(fftw_complex));
    //temporal smoothing
    for (int i = 0; i < config.halfHarmonics; i++)
    {
		float sigma = sigmaBase * powf(1. - (float)(i) / (float)config.halfHarmonics, 2.);
        float* kernel = gaussWindow(kernelSize, sigma);
	    for (int j = 0; j < effectiveLength; j++)
	    {
			(*(smoothedDftCoeffs + j * config.halfHarmonics + i))[0] = 0.;
			(*(smoothedDftCoeffs + j * config.halfHarmonics + i))[1] = 0.;
			for (int k = -(kernelSize / 2); k < kernelSize - (kernelSize / 2); k++)
			{
				int index = j + k;
				if (index < 0)
				{
					index = 0;
				}
				if (index >= effectiveLength)
				{
					index = effectiveLength - 1;
				}
				(*(smoothedDftCoeffs + j * config.halfHarmonics + i))[0] += *(kernel + k + (kernelSize / 2)) * (*(dftCoeffs + index * config.halfHarmonics + i))[0];
				(*(smoothedDftCoeffs + j * config.halfHarmonics + i))[1] += *(kernel + k + (kernelSize / 2)) * (*(dftCoeffs + index * config.halfHarmonics + i))[1];
			}
		}
        free(kernel);
	}
	for (int i = 0; i < effectiveLength * config.halfHarmonics; i++)
	{
        (*(dftCoeffs + i))[0] = (*(smoothedDftCoeffs + i))[0];
        (*(dftCoeffs + i))[1] = (*(smoothedDftCoeffs + i))[1];
	}
	free(smoothedDftCoeffs);
}

void constructVoicedSignal(fftw_complex* dftCoeffs, cSample sample, engineCfg config)
{
	int windowStart = 0;
    int windowEnd = config.tripleBatchSize;
    int markerStart = 0;
    int markerEnd = 0;
    dynIntArray windowPoints;
	dynIntArray_init(&windowPoints);
    for (int i = 0; i < sample.config.batches; i++)
    {
        windowPoints.length = 0;
        while (*(sample.pitchMarkers + markerStart) < windowStart)
        {
			if (markerStart >= sample.config.markerLength - 1)
			{
				break;
			}
            markerStart++;
        }
        markerEnd = markerStart;
        while (*(sample.pitchMarkers + markerEnd) < windowEnd)
        {
            if (markerEnd >= sample.config.markerLength - 1)
            {
                break;
            }
            dynIntArray_append(&windowPoints, markerEnd);
            markerEnd++;
        }
        if (windowPoints.length == 0)
        {
            int previous = markerStart - 1;
			if (previous < 0)
			{
				previous = 0;
			}
            int next = markerEnd;
			if (next >= sample.config.markerLength - 1)
			{
				next = sample.config.markerLength - 2;
			}
            dynIntArray_append(&windowPoints, previous);
            dynIntArray_append(&windowPoints, next);
			windowPoints.length = 2;
        }
        windowStart += config.batchSize;
        windowEnd += config.batchSize;
        for (int j = 0; j < config.halfHarmonics; j++)
        {
            float real = 0.;
		    float imag = 0.;
            float abs = 0.;
            for (int k = 0; k < windowPoints.length; k++)
            {
			    int coord = *(windowPoints.content + k);
			    real += (*(dftCoeffs + coord * config.halfHarmonics + j))[0];
			    imag += (*(dftCoeffs + coord * config.halfHarmonics + j))[1];
                abs += cpxAbsd(*(dftCoeffs + coord * config.halfHarmonics + j));
            }
			abs /= windowPoints.length;
			fftw_complex cpx = { real, imag };
			*(sample.specharm + i * config.frameSize + config.halfHarmonics - 1 - j) = abs * 2;
            *(sample.specharm + i * config.frameSize + config.nHarmonics + 1 - j) = cpxArgd(cpx);
        }
		float principalPhase = *(sample.specharm + i * config.frameSize + config.halfHarmonics + 1);
        for (int j = 0; j < config.halfHarmonics; j++)
        {
			*(sample.specharm + i * config.frameSize + config.halfHarmonics + j) -= (float)(j) * principalPhase;
			*(sample.specharm + i * config.frameSize + config.halfHarmonics + j) = fmodf(*(sample.specharm + i * config.frameSize + config.halfHarmonics + j), 2. * pi);
            if (*(sample.specharm + i * config.frameSize + config.halfHarmonics + j) > pi)
            {
                *(sample.specharm + i * config.frameSize + config.halfHarmonics + j) -= 2. * pi;
            }
			if (*(sample.specharm + i * config.frameSize + config.halfHarmonics + j) < -pi)
			{
				*(sample.specharm + i * config.frameSize + config.halfHarmonics + j) += 2. * pi;
			}
        }
    }
    dynIntArray_dealloc(&windowPoints);
}
void constructUnvoicedSignal(float* evaluationPoints, fftw_complex * dftCoeffs, float* unvoicedSignal, cSample sample, engineCfg config)
{
    //TODO: add first and last window edge cases
	for (int i = 0; i < sample.config.markerLength - 1; i++)
    {
        nfft_plan inverseNUFFT;
        int start_inner;
        int start_outer;
        if (i == 0)
		{
			start_inner = 0;
			start_outer = 0;
		}
        else
        {
			start_inner = 0.75 * *(sample.pitchMarkers + i) + 0.25 * *(sample.pitchMarkers + i + 1);
            start_outer = 0.75 * *(sample.pitchMarkers + i) + 0.25 * *(sample.pitchMarkers + i - 1);
        }
        int end_inner;
		int end_outer;
		if (i == sample.config.markerLength - 2)
		{
			end_inner = sample.config.length;
			end_outer = sample.config.length;
		}
        else
		{
			end_inner = 0.75 * *(sample.pitchMarkers + i + 1) + 0.25 * *(sample.pitchMarkers + i);
			end_outer = 0.75 * *(sample.pitchMarkers + i + 1) + 0.25 * *(sample.pitchMarkers + i + 2);
		}
		int length_inner = end_inner - start_inner;
		int length_outer = end_outer - start_outer;
        nfft_init_1d(&inverseNUFFT, config.nHarmonics, length_outer);
        for (int j = 0; j < length_outer; j++)
        {
            inverseNUFFT.x[j] = *(evaluationPoints + start_outer + j);

            if (inverseNUFFT.x[j] > 0.5)
            {
                inverseNUFFT.x[j] -= 1.;
            }
        }
        if (inverseNUFFT.flags & PRE_ONE_PSI)
        {
            nfft_precompute_one_psi(&inverseNUFFT);
        }
        for (int j = 0; j < config.halfHarmonics - 1; j++)
        {
            float abs = cpxAbsd(*(dftCoeffs + i * config.halfHarmonics + j + 1));
			float arg = cpxArgd(*(dftCoeffs + i * config.halfHarmonics + j + 1));
            inverseNUFFT.f_hat[config.nHarmonics - j - 1][0] = cos(arg) * abs;
            inverseNUFFT.f_hat[config.nHarmonics - j - 1][1] = sin(arg) * abs * -1.;
        }
        for (int j = 0; j < config.halfHarmonics; j++)
        {
            float abs = cpxAbsd(*(dftCoeffs + i * config.halfHarmonics + j));
			float arg = cpxArgd(*(dftCoeffs + i * config.halfHarmonics + j));
            inverseNUFFT.f_hat[j][0] = cos(arg) * abs;
            inverseNUFFT.f_hat[j][1] = sin(arg) * abs;
        }
        nfft_trafo_1d(&inverseNUFFT);
		for (int j = 0; j < start_inner - start_outer; j++)
		{
			*(unvoicedSignal + start_outer + j) += (*(sample.waveform + start_outer + j) - inverseNUFFT.f[j][0]) * j / (start_inner - start_outer - 1);
		}
		for (int j = 0; j < length_inner; j++)
		{
			*(unvoicedSignal + start_inner + j) += *(sample.waveform + start_inner + j) - inverseNUFFT.f[start_inner - start_outer + j][0];
		}
		for (int j = 0; j < end_outer - end_inner; j++)
		{
			*(unvoicedSignal + end_inner + j) += (*(sample.waveform + end_inner + j) - inverseNUFFT.f[end_inner - start_outer + j][0]) * (end_outer - end_inner - 1 - j) / (end_outer - end_inner - 1);
		}
        nfft_finalize(&inverseNUFFT);
    }
    //debug csv output
    /*FILE* file = fopen("unvoiced.csv", "w");
    for (int i = 0; i < sample.config.length; i++)
    {
        fprintf(file, "%f, %f\n", *(sample.waveform + i), *(unvoicedSignal + i));
    }
	fclose(file);*/
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
		return;
    }
    float* unvoicedSignal = (float*)malloc(sample.config.length * sizeof(float));
    // fill input buffer, extend data with reflection padding on both sides
    for (int i = 0; i < sample.config.length; i++)
    {
        *(unvoicedSignal + i) = 0.f;
    }
    //Get DIO Pitch markers
    fftw_complex* dftCoeffs = (fftw_complex*)malloc((sample.config.markerLength - 1) * config.halfHarmonics * sizeof(fftw_complex));
	float* evaluationPoints = (float*)malloc(sample.config.length * sizeof(float));
    float* evalPart;
	for (int i = 0; i < *(sample.pitchMarkers); i++)
	{
		*(evaluationPoints + i) = (float)i / (float)*(sample.pitchMarkers);
	}
    if (sample.config.markerLength <= 8)
    {
		evalPart = getEvaluationPoints(0, sample.config.markerLength - 1, sample.waveform, sample, config);
        for (int i = *(sample.pitchMarkers); i < *(sample.pitchMarkers + sample.config.markerLength - 1); i++)
        {
            *(evaluationPoints + i) = fmodf(*(evalPart + i - *(sample.pitchMarkers)), 1.);
        }
		free(evalPart);
    }
    else
    {
        evalPart = getEvaluationPoints(0, 8, sample.waveform, sample, config);
        for (int i = *(sample.pitchMarkers); i < *(sample.pitchMarkers + 7); i++)
        {
            *(evaluationPoints + i) = fmodf(*(evalPart + i - *(sample.pitchMarkers)), 1.);
        }
        free(evalPart);
        for (int i = 6; i < sample.config.markerLength; i += 6)
        {
            if (i >= sample.config.markerLength - 9)
            {
                break;
            }
            evalPart = getEvaluationPoints(i, i + 8, sample.waveform, sample, config);
            for (int j = *(sample.pitchMarkers + i + 1); j < *(sample.pitchMarkers + i + 7); j++)
            {
                *(evaluationPoints + j) = fmodf(*(evalPart + j - *(sample.pitchMarkers + i + 1)), 1.);
            }
            free(evalPart);
        }
        evalPart = getEvaluationPoints(sample.config.markerLength - 9, sample.config.markerLength - 1, sample.waveform, sample, config);
        for (int i = *(sample.pitchMarkers + sample.config.markerLength - 8); i < *(sample.pitchMarkers + sample.config.markerLength - 1); i++)
        {
            *(evaluationPoints + i) = fmodf(*(evalPart + i - *(sample.pitchMarkers + sample.config.markerLength - 8)), 1.);
        }
        free(evalPart);
    }
	int lastMarker = *(sample.pitchMarkers + sample.config.markerLength - 1);
	int postLength = sample.config.length - *(sample.pitchMarkers + sample.config.markerLength - 1);
	for (int i = 0; i < postLength; i++)
	{
		*(evaluationPoints + lastMarker + i) = (float)i / (float)postLength;
	}
    for (int i = 0; i < sample.config.markerLength - 1; i++)
    {
        separateVoicedUnvoicedSingleWindow(i, evaluationPoints, dftCoeffs, sample, config);
    }
    separateVoicedUnvoicedPostProc(dftCoeffs, sample, config);
	constructVoicedSignal(dftCoeffs, sample, config);
	constructUnvoicedSignal(evaluationPoints, dftCoeffs, unvoicedSignal, sample, config);
	free(evaluationPoints);
    free(dftCoeffs);
    stft_inpl(unvoicedSignal, sample.config.length, config, sample.excitation);
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
    return;
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