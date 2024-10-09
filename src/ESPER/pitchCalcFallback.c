//Copyright 2023 Johannes Klatt

//This file is part of libESPER.
//libESPER is free software: you can redistribute it and /or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
//libESPER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with Nova - Vox.If not, see <https://www.gnu.org/licenses/>.

#include "src/ESPER/pitchCalcFallback.h"

#include <malloc.h>
#include <math.h>
#include "src/util.h"

#include <Windows.h>

typedef struct
{
    unsigned int position;
	void* previous;
	float distance;
	int isRoot;
	int isLeaf;
} MarkerCandidate;

//fallback function for calculating the approximate time-dependent pitch of a sample.
//Used when the Torchaudio implementation fails, likely due to a too narrow search range setting or the sample being too short.
void LIBESPER_CDECL pitchCalcFallback(cSample* sample, engineCfg config) {
	//limits for autocorrelation search
	unsigned int batchSize;
	unsigned int lowerLimit;
	if (sample->config.expectedPitch == 0) {
		batchSize = config.tripleBatchSize;
		lowerLimit = config.sampleRate / 1000;
	}
	else
	{
		batchSize = (int)((1. + sample->config.searchRange) * (float)config.sampleRate / (float)sample->config.expectedPitch);
		if (batchSize > config.tripleBatchSize)
		{
			batchSize = config.tripleBatchSize;
		}
		lowerLimit = (int)((1. - sample->config.searchRange) * (float)config.sampleRate / (float)sample->config.expectedPitch);
		if (lowerLimit < config.sampleRate / 1000)
		{
			lowerLimit = config.sampleRate / 1000;
		}
	}
	/*
	unsigned int batchStart = 0;
	//oversized array for holding all zeroTransitions within a batch
	unsigned int* zeroTransitions = (unsigned int*) malloc(batchSize * sizeof(unsigned int));
	unsigned int numZeroTransitions;
	double error;
	unsigned int delta;
	float bias;
	unsigned int offset;
	*/
	//buffer for storing pitch in pitch-synchronous format
	//unsigned int* intermediateBuffer = (unsigned int*) malloc(ceildiv(sample->config.length, lowerLimit) * sizeof(unsigned int));
	//unsigned int intermBufferLen = 0;


	float* smoothedWave = (float*)malloc(sample->config.length * sizeof(float));
	float x = 0;
	float v = 0;
	float a = 0;
	for (int i = 0; i < sample->config.length; i++) {
		a = *(sample->waveform + i) - 0.1 * x - 0.1 * v;
		v += a;
		x += v;
		*(smoothedWave + i) = x;
	}

	dynIntArray zeroTransitions;
	dynIntArray_init(&zeroTransitions);
	for (int i = 1; i < sample->config.length; i++) {
		if (*(smoothedWave + i - 1) < 0 && *(smoothedWave + i) >= 0) {
			dynIntArray_append(&zeroTransitions, i);
		}
	}
	int markerCandidateLength = zeroTransitions.length;
	MarkerCandidate* markerCandidates = (MarkerCandidate*)malloc(zeroTransitions.length * sizeof(MarkerCandidate));
	for (int i = 0; i < zeroTransitions.length; i++) {
		(markerCandidates + i)->position = zeroTransitions.content[i];
		(markerCandidates + i)->previous = NULL;
		(markerCandidates + i)->distance = 0;
		if (zeroTransitions.content[i] < batchSize || i == 0) {
			(markerCandidates + i)->isRoot = 1;
		}
		else
		{
			(markerCandidates + i)->isRoot = 0;
		}
		if (zeroTransitions.content[i] >= sample->config.length - batchSize || i == zeroTransitions.length - 1) {
			(markerCandidates + i)->isLeaf = 1;
		}
		else
		{
			(markerCandidates + i)->isLeaf = 0;
		}
	}

	for (int i = 0; i < markerCandidateLength; i++)
	{
		if ((markerCandidates + i)->isRoot)
		{
			continue;
		}
		int isValid = 0;
		for (int j = 1; j <= i; j++)
		{
			unsigned int positionI = (markerCandidates + i)->position;
			unsigned int positionJ = (markerCandidates + i - j)->position;
			unsigned int delta = positionI - positionJ;
			if (delta < lowerLimit)
			{
				continue;
			}
			if ((delta > batchSize) && isValid == 1)
			{
				break;
			}
			float bias;
			if (sample->config.expectedPitch == 0)
			{
				bias = 1.;
			}
			else
			{
				bias = fabsf(delta - (float)config.sampleRate / (float)sample->config.expectedPitch);
			}
			double newError = 0;
			double contrast = 0;
			if (positionJ < batchSize)
			{
				for (int k = 0; k < delta; k++)
				{
					newError += powf(*(smoothedWave + positionI + k) - *(smoothedWave + positionJ + k), 2.) * bias;
					contrast += *(smoothedWave + positionI + k) * sinf(2. * pi * k / delta);
				}
			}
			else if (positionI >= sample->config.length - batchSize)
			{
				for (int k = 0; k < delta; k++)
				{
					newError += powf(*(smoothedWave + positionI - k) - *(smoothedWave + positionJ - k), 2.) * bias;
					contrast += *(smoothedWave + positionI - k) * sinf(2. * pi * k / delta);
				}
			}
			else
			{
				for (int k = 0; k < delta; k++)
				{
					newError += powf(*(smoothedWave + positionI - delta / 2 + k) - *(smoothedWave + positionJ - delta / 2 + k), 2.) * bias;
					contrast += *(smoothedWave + positionI - delta / 2 + k) * sinf(2. * pi * k / delta);
				}
			}

			if ((markerCandidates + i - j)->distance + newError / powf(contrast, 2.) < (markerCandidates + i)->distance || (markerCandidates + i)->distance == 0) {
				(markerCandidates + i)->distance = (markerCandidates + i - j)->distance + newError / powf(contrast, 2.);
				(markerCandidates + i)->previous = markerCandidates + i - j;
			}
			isValid = 1;
		}
	}
	zeroTransitions.length = 0;
	MarkerCandidate* currentBase = markerCandidates + markerCandidateLength - 1;
	MarkerCandidate* current = currentBase;
	while (currentBase->isLeaf)
	{
		if (currentBase->distance < current->distance)
		{
			current = currentBase;
		}
		currentBase--;
	}
	while (current->previous != NULL)
	{
		dynIntArray_append(&zeroTransitions, current->position);
		current = current->previous;
	}
	sample->config.markerLength = zeroTransitions.length;
	for (int i = 0; i < zeroTransitions.length; i++) {
		*(sample->pitchMarkers + i) = zeroTransitions.content[zeroTransitions.length - i - 1];
	}
	dynIntArray_dealloc(&zeroTransitions);
	free(markerCandidates);
	free(smoothedWave);

	unsigned int cursor = 0;
	for (int i = 0; i < sample->config.batches; i++) {
		while ((sample->pitchMarkers[cursor] <= i * config.batchSize) && (cursor < sample->config.markerLength)) {
			cursor++;
		}
		if (cursor == 0) {
			*(sample->pitchDeltas + i) = sample->pitchMarkers[cursor + 1] - sample->pitchMarkers[cursor];
		}
		else if (cursor == sample->config.markerLength) {
			*(sample->pitchDeltas + i) = sample->pitchMarkers[cursor - 1] - sample->pitchMarkers[cursor - 2];
		}
		else
		{
			*(sample->pitchDeltas + i) = sample->pitchMarkers[cursor] - sample->pitchMarkers[cursor - 1];
		}
	}
	sample->config.pitch = median(sample->pitchDeltas, sample->config.pitchLength);
	for (int i = 0; i < sample->config.markerLength; i++) {
		*(sample->pitchMarkers + i) += config.halfTripleBatchSize * config.filterBSMult;
	}
}


































    //run until sample is fully processed
    /*
    while (batchStart + batchSize <= sample->config.length - batchSize) {
        //get all zero-transitions in the current batch
        numZeroTransitions = 0;
        for (int i = batchStart + lowerLimit; i < batchStart + batchSize; i++) {
            if ((*(smoothedWave + i - 1) < 0) && (*(smoothedWave + i) > 0)) {
                *(zeroTransitions + numZeroTransitions) = i;
                numZeroTransitions++;
            }
        }
        //calculate autocorrelation error for each zeroTransition; load the value of the best candidate into delta
        error = -1;
		if (sample->config.expectedPitch == 0) {
			delta = 300;
		}
		else
		{
			delta = config.sampleRate / sample->config.expectedPitch;
		}
        for (int i = 0; i < numZeroTransitions; i++) {
            offset = *(zeroTransitions + i);
			if (sample->config.expectedPitch == 0) {
				bias = 1.;
			}
            else
            {
                bias = fabsf(offset - batchStart - (float)config.sampleRate / (float)sample->config.expectedPitch);
			}
            double newError = 0;
			double contrast = 0;
            for (int j = 0; j < batchSize; j++) {
                newError += powf(*(smoothedWave + batchStart + j) - *(smoothedWave + offset + j), 2.) * bias;
            }
			for (int j = 0; j < batchSize - batchSize % (offset - batchStart); j++) {
                contrast += *(smoothedWave + batchStart + j) * sinf(2. * pi * j / (offset - batchStart));
			}
			contrast /= batchSize - batchSize % (offset - batchStart);
            newError /= fabsf(contrast);
            newError *= (float)(offset - batchStart) / (float)batchSize;
            if ((error > newError) || (error == -1)) {
                delta = offset - batchStart;
                error = newError;
            }
        }
        //add best candidate to buffer and use it as starting point for the next batch
        *(intermediateBuffer + intermBufferLen) = delta;
        intermBufferLen++;
        batchStart += delta;
    }
    free(zeroTransitions);

	//reverse pass to improve accuracy
    





    unsigned int cursor = 0;
    int cursor2 = 0;
    //calculate "virtual" pitch-based sample length
    sample->config.pitchLength = 0;
    for (int i = 0; i < sample->config.markerLength; i++) {
        sample->config.pitchLength += *(sample->pitchMarkers + i);
    }
    sample->config.pitchLength /= config.batchSize;
    sample->config.pitch /= sample->config.markerLength;
    if (sample->config.pitchLength > sample->config.batches)
    {
        sample->config.pitchLength = sample->config.batches;
    }
    //transform from pitch-synchronous to time-synchronous space and fill average pitch field
    for (int i = 0; i < sample->config.pitchLength; i++) {
        while(cursor2 >= (int)*(sample->pitchMarkers + cursor)) {
            if (cursor < sample->config.markerLength - 1) {
                cursor++;
            }
            cursor2 -= *(sample->pitchMarkers + cursor);
        }
        cursor2 += config.batchSize;
        *(sample->pitchDeltas + i) = *(sample->pitchMarkers + cursor);
    }
	sample->config.pitch = median(sample->pitchMarkers, sample->config.markerLength);
}*/


