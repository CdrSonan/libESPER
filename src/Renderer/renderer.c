//Copyright 2024 Johannes Klatt

//This file is part of libESPER.
//libESPER is free software: you can redistribute it and /or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
//libESPER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with Nova - Vox.If not, see <https://www.gnu.org/licenses/>.

#include "src/Renderer/renderer.h"

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include "src/util.h"
#include "src/fft.h"
#include "src/interpolation.h"
#include LIBESPER_FFTW_INCLUDE_PATH
#include LIBESPER_NFFT_INCLUDE_PATH

void LIBESPER_CDECL renderUnvoiced(float* specharm, float* excitation, int premultiplied, float* target, int length, engineCfg config)
{
	fftwf_complex* cpxExcitation = (fftwf_complex*)malloc(length * (config.halfTripleBatchSize + 1) * sizeof(fftwf_complex));
	if (premultiplied != 0)
	{
		for (int i = 0; i < length * (config.halfTripleBatchSize + 1); i++)
		{
			(*(cpxExcitation + i))[0] = *(excitation + i);
			(*(cpxExcitation + i))[1] = *(excitation + length * (config.halfTripleBatchSize + 1) + i);
		}
	}
	else
	{
		for (int i = 0; i < length; i++)
		{
			for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
			{
				float multiplier = *(specharm + i * config.frameSize + config.nHarmonics + 2 + i);
				(*(cpxExcitation + i))[0] = *(excitation + i) * multiplier;
				(*(cpxExcitation + i))[1] = *(excitation + length * (config.halfTripleBatchSize + 1) + i) * multiplier;
			}
		}
	}
	istft_hann_inpl(cpxExcitation, length, length * config.batchSize, config, target);
	free(cpxExcitation);
}

void LIBESPER_CDECL renderVoiced(float* specharm, float* pitch, float* phase, float* target, int length, engineCfg config)
{
	float* frameSpace = (float*)malloc(length * sizeof(float));
	for (int i = 0; i < length; i++)
	{
		*(frameSpace + i) = (float)i;
	}
	float* waveSpace = (float*)malloc(length * config.batchSize * sizeof(float));
	for (int i = 0; i < length * config.batchSize; i++)
	{
		*(waveSpace + i) = ((float)i - 0.5) / (float)config.batchSize;
	}
	float* wavePitch = extrap(frameSpace, pitch, waveSpace, length, length * config.batchSize);
	float* pitchOffsets = (float*)malloc(length * config.batchSize * sizeof(float));
	*pitchOffsets = 0.;
	for (int i = 0; i < length * config.batchSize - 1; i++)
	{
		*(pitchOffsets + i + 1) = *(pitchOffsets + i) + 1. / *(wavePitch + i);
	}
	free(wavePitch);
	float* harmAbs = (float*)malloc(length * sizeof(float));
	float* harmArg = (float*)malloc(length * sizeof(float));
	float* interpAbs = (float*)malloc(length * config.batchSize * sizeof(float));
	float* interpArg = (float*)malloc(length * config.batchSize * sizeof(float));
	for (int i = 0; i < config.halfHarmonics; i++)
	{
		for (int j = 0; j < length; j++)
		{
			*(harmAbs + j) = *(specharm + j * config.frameSize + i);
			*(harmArg + j) = *(specharm + j * config.frameSize + config.halfHarmonics + i);
		}
		extrap_inpl(frameSpace, harmAbs, waveSpace, length, length * config.batchSize, interpAbs);
		circInterp_inpl(frameSpace, harmArg, waveSpace, length, length * config.batchSize, interpArg);
		for (int j = 0; j < length * config.batchSize - 1; j++)
		{
			*(target + j) += cos(*(interpArg + j) + *(pitchOffsets + j) * 2. * pi * i) * *(interpAbs + j);
		}
	}
	free(harmAbs);
	free(harmArg);
	free(interpAbs);
	free(interpArg);
	free(pitchOffsets);
	free(frameSpace);
	free(waveSpace);
}

void LIBESPER_CDECL renderVoiced_experimental(float* specharm, float* pitch, float* phase, float* target, int length, engineCfg config)
{
	float* frameSpace = (float*)malloc(length * sizeof(float));
	for (int i = 0; i < length; i++)
	{
		*(frameSpace + i) = (float)i;
	}
	float* waveSpace = (float*)malloc((length + 2) * config.batchSize * sizeof(float));
	for (int i = 0; i < length * config.batchSize; i++)
	{
		*(waveSpace + i) = ((float)i - 0.5 - config.batchSize) / (float)config.batchSize;
	}
	float* wavePitch = extrap(frameSpace, pitch, waveSpace, length, (length + 2) * config.batchSize);
	free(frameSpace);
	free(waveSpace);
	float* evaluationPoints = (float*)malloc((length + 2) * config.batchSize * sizeof(float));
	*(evaluationPoints + config.batchSize) = *phase;
	for (int i = 0; i < config.batchSize; i++)
	{
		*(evaluationPoints + config.batchSize - i - 1) = *(evaluationPoints + config.batchSize - i) - 1. / *(wavePitch + config.batchSize - i - 1);
		*(evaluationPoints + config.batchSize - i - 1) = fmodf(*(evaluationPoints + config.batchSize - i - 1), 1.);
	}
	for (int i = config.batchSize; i < (length + 2) * config.batchSize - 1; i++)
	{
		*(evaluationPoints + i + 1) = *(evaluationPoints + i) + 1. / *(wavePitch + i);
		*(evaluationPoints + i + 1) = fmodf(*(evaluationPoints + i + 1), 1.);
	}
	*phase = *(evaluationPoints + (length + 1) * config.batchSize);
	free(wavePitch);
	float* hannWindowInst = hannWindow(config.tripleBatchSize, 1.);
	nfft_plan inverseNUFFT;
	nfft_init_1d(&inverseNUFFT, config.nHarmonics, config.tripleBatchSize);
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < config.tripleBatchSize; j++)
		{
			inverseNUFFT.x[j] = fmodf(*(evaluationPoints + i * config.batchSize + j), 1.f);
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
			inverseNUFFT.f_hat[config.nHarmonics - j - 1][0] = cos(*(specharm + i * config.frameSize + config.nHarmonics - j)) * *(specharm + i * config.frameSize + config.halfHarmonics - 2 - j);
			inverseNUFFT.f_hat[config.nHarmonics - j - 1][1] = sin(*(specharm + i * config.frameSize + config.nHarmonics - j)) * *(specharm + i * config.frameSize + config.halfHarmonics - 2 - j) * -1.;
		}
		for (int j = 0; j < config.halfHarmonics; j++)
		{
			inverseNUFFT.f_hat[j][0] = cos(*(specharm + i * config.frameSize + config.nHarmonics + 1 - j)) * *(specharm + i * config.frameSize + config.halfHarmonics - 1 - j);
			inverseNUFFT.f_hat[j][1] = sin(*(specharm + i * config.frameSize + config.nHarmonics + 1 - j)) * *(specharm + i * config.frameSize + config.halfHarmonics - 1 - j);
		}
		nfft_trafo_1d(&inverseNUFFT);
		int lowerLimit;
		int upperLimit;
		if (i == 0)
		{
			lowerLimit = config.batchSize;
		}
		else
		{
			lowerLimit = 0;
		}
		if (i == length - 1)
		{
			upperLimit = 2 * config.batchSize;
		}
		else
		{
			upperLimit = config.tripleBatchSize;
		}
		for (int j = lowerLimit; j < upperLimit; j++)
		{
			*(target + i * config.batchSize + j - config.batchSize) +=  inverseNUFFT.f[j][0] * *(hannWindowInst + j);
		}
	}
	nfft_finalize(&inverseNUFFT);
	free(evaluationPoints);
}

void LIBESPER_CDECL render(float* specharm, float* excitation, float* pitch, int premultipliedExc, float* phase, float* target, int length, engineCfg config)
{
	renderUnvoiced(specharm, excitation, premultipliedExc, target, length, config);
	renderVoiced(specharm, pitch, phase, target, length, config);
}
