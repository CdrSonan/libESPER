#pragma once

#include "src/util.h"

void loopSamplerSpecharm(float* input, int length, float* output, int targetLength, float spacing, engineCfg config);

void loopSamplerPitch(short* input, int length, float* output, int targetLength, float spacing);
