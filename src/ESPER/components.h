#pragma once

#include "src/util.h"

float* lowRangeSmooth(cSample sample, float* signalsAbs, engineCfg config);

float* highRangeSmooth(cSample sample, float* signalsAbs, engineCfg config);

void finalizeSpectra(cSample sample, float* lowSpectra, float* highSpectra, engineCfg config);

void separateVoicedUnvoiced(cSample sample, engineCfg config);

void averageSpectra(cSample sample, engineCfg config);
