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

float* createSmoothedWave(cSample* sample)
{
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
	return smoothedWave;
}

dynIntArray getZeroTransitions(float* signal, int length)
{
	dynIntArray zeroTransitions;
	dynIntArray_init(&zeroTransitions);
	for (int i = 1; i < length; i++) {
		if (*(signal + i - 1) < 0 && *(signal + i) >= 0) {
			dynIntArray_append(&zeroTransitions, i);
		}
	}
	return zeroTransitions;
}

MarkerCandidate* createMarkerCandidates(dynIntArray zeroTransitions, unsigned int batchSize, unsigned int lowerLimit, unsigned int length)
{
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
		if (zeroTransitions.content[i] >= length - batchSize || i == zeroTransitions.length - 1) {
			(markerCandidates + i)->isLeaf = 1;
		}
		else
		{
			(markerCandidates + i)->isLeaf = 0;
		}
	}
	return markerCandidates;
}

void buildPitchGraph(MarkerCandidate* markerCandidates, int markerCandidateLength, unsigned int batchSize, unsigned int lowerLimit, cSample* sample, float* smoothedWave, engineCfg config)
{
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
}

void fillPitchMarkers(dynIntArray zeroTransitions, MarkerCandidate* markerCandidates, int markerCandidateLength, cSample* sample)
{
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
}

void fillPitchDeltas(cSample* sample, engineCfg config)
{
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
}

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
	float* smoothedWave = createSmoothedWave(sample);
	dynIntArray zeroTransitions = getZeroTransitions(smoothedWave, sample->config.length);
	int markerCandidateLength = zeroTransitions.length;
	MarkerCandidate* markerCandidates = createMarkerCandidates(zeroTransitions, batchSize, lowerLimit, sample->config.length);
	buildPitchGraph(markerCandidates, markerCandidateLength, batchSize, lowerLimit, sample, smoothedWave, config);
	fillPitchMarkers(zeroTransitions, markerCandidates, markerCandidateLength, sample);
	dynIntArray_dealloc(&zeroTransitions);
	free(markerCandidates);
	free(smoothedWave);
	fillPitchDeltas(sample, config);
	sample->config.pitch = median(sample->pitchDeltas, sample->config.pitchLength);
}
