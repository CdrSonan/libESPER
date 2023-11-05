#pragma once

#include "src/util.h"

__declspec(dllexport) void __cdecl resampleSpecharm(float* avgSpecharm, float* specharm, int length, float* steadiness, float spacing, int startCap, int endCap, float* output, segmentTiming timings, engineCfg config);

__declspec(dllexport) void __cdecl resamplePitch(int* pitchDeltas, int length, float pitch, float spacing, int startCap, int endCap, float* output, int requiredSize, segmentTiming timings);
