#pragma once

float* interp(float* x, float* y, float* xs, int len, int lenxs);

float* extrap(float* x, float* y, float* xs, int len, int lenxs);

void phaseInterp_inplace(float* phasesA, float* phasesB, int len, float factor);