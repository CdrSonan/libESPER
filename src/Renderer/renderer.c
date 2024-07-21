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

void LIBESPER_CDECL renderVoiced(float* specharm, float* pitch, float* target, int length, engineCfg config)
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
	for (int i = 0; i < config.halfHarmonics; i++)
	{
		float* harmAbs = (float*)malloc(length * sizeof(float));
		float* harmArg = (float*)malloc(length * sizeof(float));
		for (int j = 0; j < length; j++)
		{
			*(harmAbs + j) = *(specharm + j * config.frameSize + i);
			*(harmArg + j) = *(specharm + j * config.frameSize + config.halfHarmonics + i);
		}
		float* interpAbs = extrap(frameSpace, harmAbs, waveSpace, length, length * config.batchSize);
		float* interpArg = circInterp(frameSpace, harmArg, waveSpace, length, length * config.batchSize);
		free(harmAbs);
		free(harmArg);
		for (int j = 0; j < length * config.batchSize - 1; j++)
		{
			*(target + j) += cos(*(interpArg + j) + *(pitchOffsets + j) * 2. * pi * i) * *(interpAbs + j);
		}
	}
	free(pitchOffsets);
	free(frameSpace);
	free(waveSpace);
}

void LIBESPER_CDECL render(float* specharm, float* excitation, float* pitch, int premultipliedExc, float* target, int length, engineCfg config)
{
	renderUnvoiced(specharm, excitation, premultipliedExc, target, length, config);
	renderVoiced(specharm, pitch, target, length, config);
}
