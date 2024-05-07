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

void LIBESPER_CDECL render(float* specharm, float* excitation, float* pitch, float* target, int length, engineCfg config)
{
	printf("C input params: %p %p %p %p %i\n", specharm, excitation, pitch, target, length);
	printf("C render 1\n");
	//fftwf_complex* cpxExcitation = (fftwf_complex*)malloc(length * (config.halfTripleBatchSize + 1) * sizeof(fftwf_complex));
	//for (int i = 0; i < length * (config.halfTripleBatchSize + 1); i++)
	//{
	//	*(cpxExcitation + i)[0] = 0.;//*(excitation + i);
	//	*(cpxExcitation + i)[1] = 0.;//*(excitation + length * (config.halfTripleBatchSize + 1) + i);
	//}
	//printf("C render 2\n");
	//istft_hann_inpl(cpxExcitation, length, length * config.batchSize, config, target);
	printf("C render 3\n");
	//free(cpxExcitation);
	float* frameSpace = (float*)malloc(length * sizeof(float));
	for (int i = 0; i < length; i++)
	{
		*(frameSpace + i) = (float)i;
	}
	printf("C render 4\n");
	float* waveSpace = (float*)malloc(length * config.batchSize * sizeof(float));
	for (int i = 0; i < length * config.batchSize; i++)
	{
		*(waveSpace + i) = ((float)i - 0.5) / (float)config.batchSize;
	}
	printf("C render 5\n");
	float* wavePitch = extrap(frameSpace, pitch, waveSpace, length, length * config.batchSize);
	printf("C render 6\n");
	float* pitchOffsets = (float*)malloc(length * config.batchSize * sizeof(float));
	*pitchOffsets = 0.;
	for (int i = 0; i < length * config.batchSize - 1; i++)
	{
		*(pitchOffsets + i + 1) = *(pitchOffsets + i) + 1. / *(wavePitch + i);
	}
	free(wavePitch);
	printf("C render 7\n");
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
	printf("C render 8\n");
}
