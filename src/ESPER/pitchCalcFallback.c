//Copyright 2023 Johannes Klatt

//This file is part of libESPER.
//libESPER is free software: you can redistribute it and /or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
//libESPER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with Nova - Vox.If not, see <https://www.gnu.org/licenses/>.

#include "src/ESPER/pitchCalcFallback.h"

#include <malloc.h>
#include <math.h>
#include "src/util.h"

//defines a struct for a marker candidate in the pitch calculation process.
//attributes:
// - position: the position of the zero crossing in the signal.
// - previous: a pointer to the previous marker candidate in the pitch graph. Filled using Dijkstra's algorithm.
// - distance: the distance from the nearest root marker candidate to this marker candidate in the pitch graph. Filled using Dijkstra's algorithm.
// - isRoot: a flag indicating whether this marker candidate is a root marker candidate. Root marker candidates have no predecessors in the pitch graph.
// - isLeaf: a flag indicating whether this marker candidate is a leaf marker candidate. Leaf marker candidates have no successors in the pitch graph.
typedef struct
{
    unsigned int position;
	void* previous;
	float distance;
	int isRoot;
	int isLeaf;
} MarkerCandidate;

//uses a momentum function to smooth the waveform of a sample.
//this is done to remove high-frequency noise and lower the effect of volume changes by dampening regions with high audio volume.
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

//returns an array of upwards zero crossings in a signal.
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

//converts an array of zero crossings to a corresponding array of MarkerCandidate structs.
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

//builds a pitch graph using an adapted version of Dijkstra's algorithm.
//The nodes of the graph are the zero crossings in the signal.
//An edge is assumed to exist between two nodes if their distance in signal space is above the lowerLimit argument, but below the batchSize argument.
//Edges always point from a node to a node with a higher index, making the graph a directed acyclic graph.
//If a node has no outgoing edges according to these rules, an edge connecting it to the next node in signal space is assumed to exist.
//The weight of each edge is calculated through two objectives:
// - minimizing the squared difference between the signal values in the vicinity of the two nodes
// - maximizing the amplitude of the assumed f0 component of the signal. This is achieved by multiplying the signal in the vicinity of the first node with a sine wave of the frequency corresponding to the distance between the two nodes.
//The first term ensures that the pitch period is not underestimated, since the similarity between different parts of the same signal period is low.
//The second term prevents the pitch period from being overestimated as a multiple of the real pitch period, since multiple repetitions of the same signal period in sequence do not have a frequency component matching the assumed f0.
//
//Based on this, a modified version of Dijkstra's algorithm is used to find the shortest path from any root node to all other nodes in the graph.
//The modification is simple: Instead of starting from a single root, multiple are specified, each with distance 0 and no predecessors. The algorithm then proceeds as usual.
//Additionally, since we already know an ordering of the nodes, all nodes can be evaluated in sequence, omitting the open set and closed set arrays used in the original algorithm.
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

//takes a pitch graph, given as an array of MarkerCandidate structs, and fills an array of pitch markers with the positions of the zero crossings corresponding to the optimal path through the graph.
//The optimal path is determined by finding the leaf node with the lowest distance to any root node, and then following the previous pointers.
//Since this gives the path in reverse order, the zeroTransitions array is reused to intermittently store the reversed path, before copying it to the pitchMarkers array in the correct order.
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

//fills an array of pitch deltas with the differences between the pitch markers.
//pitch markers represent the pitch curve in pitch-synchronous form, while the pitch deltas represent the pitch at constant time intervals.
//Therefore, the pitch deltas are calculated by finding the pitch markers closest to the current time interval, and taking the difference between them.
//This is a sampling operation, so there is no guarantee all pitch markers will be used. Likewise, in extreme cases, the same markers may also be used several times.
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

//filters the pitch deltas using a median filter with a window size of 5.
void filterPitchDeltas(cSample* sample)
{
	int* medianBuffer = (int*)malloc(5 * sizeof(int));
	int* filteredDeltas = (int*)malloc(sample->config.batches * sizeof(int));
	for (int i = 0; i < sample->config.batches; i++)
	{
		if (i < 2 || i > sample->config.batches - 3)
		{
			*(filteredDeltas + i) = *(sample->pitchDeltas + i);
		}
		else
		{
			for (int j = 0; j < 5; j++) {
				*(medianBuffer + j) = *(sample->pitchDeltas + i - 2 + j);
			}
			*(filteredDeltas + i) = median(medianBuffer, 5);
		}
	}
	free(medianBuffer);
	for (int i = 0; i < sample->config.batches; i++)
	{
		*(sample->pitchDeltas + i) = *(filteredDeltas + i);
	}
	free(filteredDeltas);
}

//master function for pitch calculation.
//Fills the pitchMarkers and pitchDeltas arrays of a sample, representing the pitch in pitych-synchronous and constant time intervals, respectively, and calculates the median pitch.
//The name "fallback" is left over from a previous version, where it was only used if pitch calculation using torchaudio failed.
//It is now the only pitch calculation method, as other methods cannot provide the pitch-synchronous representation required by other parts of the library with sufficient accuracy.
void LIBESPER_CDECL pitchCalcFallback(cSample* sample, engineCfg config) {

	//limits for pitch graph edge creation
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
	filterPitchDeltas(sample);
	sample->config.pitch = median(sample->pitchDeltas, sample->config.pitchLength);
}
