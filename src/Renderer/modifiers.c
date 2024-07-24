//Copyright 2024 Johannes Klatt

//This file is part of libESPER.
//libESPER is free software: you can redistribute it and /or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
//libESPER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with Nova - Vox.If not, see <https://www.gnu.org/licenses/>.

#include "src/Renderer/modifiers.h"

#include <malloc.h>
#include <math.h>
#include <stdio.h>
#include "src/util.h"
#include "src/fft.h"
#include "src/interpolation.h"
#include LIBESPER_FFTW_INCLUDE_PATH

void applyBreathiness(float* specharm, float* breathiness, int length, engineCfg config)
{
	for (int i = 0; i < length; i++)
	{
		float compensation = 0.;
		float divisor = 1.;
		for (int j = 0; j < config.halfHarmonics; j++)
		{
			compensation += powf(*(specharm + i * config.frameSize + j), 2.);
		}
		for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
		{
			divisor += powf(*(specharm + i * config.frameSize + config.nHarmonics + 2 + j), 2.);
		}
		compensation *= config.breCompPremul / divisor;
		float breathinessVoiced;
		float breathinessUnvoiced;
		if (*(breathiness + i) >= 0.)
		{
			breathinessVoiced = 1. - *(breathiness + i);
			breathinessUnvoiced = 1. + *(breathiness + i) * compensation;
		}
		else
		{
			breathinessVoiced = 1.;
			breathinessUnvoiced = *(breathiness + i);
		}
		for (int j = 0; j < config.halfHarmonics; j++)
		{
			*(specharm + i * config.frameSize + j) *= breathinessVoiced;
		}
		for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
		{
			*(specharm + i * config.frameSize + config.nHarmonics + 2 + j) *= breathinessUnvoiced;
		}
	}
}

void pitchShift(float* specharm, float* srcPitch, float* tgtPitch, float* formantShift, float* breathiness, int length, engineCfg config)
{
	float* harmonicsSpace = (float*)malloc(config.halfHarmonics * sizeof(float));
	float* spectrumSpace = (float*)malloc((config.halfTripleBatchSize + 1) * sizeof(float));
	for (int i = 0; i < config.halfTripleBatchSize + 1; i++)
	{
		*(spectrumSpace + i) = i;
	}
	float* multipliers = (float*)malloc((config.halfTripleBatchSize + 1) * sizeof(float));
	float* shiftedSpectrumSpace = (float*)malloc((config.halfTripleBatchSize + 1) * sizeof(float));
	for (int i = 0; i < length; i++)
	{
		float effSrcPitch = (float)config.tripleBatchSize / *(srcPitch + i);
		float effTgtPitch = (float)config.tripleBatchSize / *(tgtPitch + i);

		for (int j = 0; j < config.halfHarmonics; j++)
		{
			*(harmonicsSpace + j) = j * effSrcPitch;
		}
		interp_inpl(
			spectrumSpace,
			specharm + i * config.frameSize + config.nHarmonics + 2,
			harmonicsSpace,
			config.halfTripleBatchSize + 1,
			config.halfHarmonics,
			multipliers);
		for (int j = 0; j < config.halfHarmonics; j++)
		{
			*(specharm + i * config.frameSize + j) /= *(multipliers + j);
		}

		for (int j = 0; j < config.halfHarmonics; j++)
		{
			*(harmonicsSpace + j) = j * effTgtPitch;
		}
		interp_inpl(
			spectrumSpace,
			specharm + i * config.frameSize + config.nHarmonics + 2,
			harmonicsSpace,
			config.halfTripleBatchSize + 1,
			config.halfHarmonics,
			multipliers);
		for (int j = 0; j < config.halfHarmonics; j++)
		{
			*(specharm + i * config.frameSize + j) *= *(multipliers + j);
		}


		if (*(breathiness + i) <= 0)
		{
			continue;
		}
		effTgtPitch = *(breathiness + i) * effTgtPitch + (1. - *(breathiness + i)) * effSrcPitch;
		for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
		{
			*(shiftedSpectrumSpace + j) = j * effTgtPitch;
			if (*(shiftedSpectrumSpace + j) >= config.halfTripleBatchSize + 1)
			{
				*(shiftedSpectrumSpace + j) = config.halfTripleBatchSize;
			}
		}
		interp_inpl(
			spectrumSpace,
			specharm + i * config.frameSize + config.nHarmonics + 2,
			shiftedSpectrumSpace,
			config.halfTripleBatchSize + 1,
			config.halfTripleBatchSize + 1,
			multipliers);
		for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
		{
			*(specharm + i * config.frameSize + config.nHarmonics + 2 + j) *= *(multipliers + j);
		}
	}
	free(harmonicsSpace);
	free(spectrumSpace);
	free(multipliers);
	free(shiftedSpectrumSpace);
}

void applyPeakDampening(float* specharm, float* compression, int length, engineCfg config)
{
	for (int i = 0; i < length; i++)
	{
		for (int j = 0; j < config.halfHarmonics; j++)
		{
			*(specharm + i * config.frameSize + j) *= breathinessVoiced;
		}
		for (int j = 0; j < config.halfTripleBatchSize + 1; j++)
		{
			*(specharm + i * config.frameSize + config.nHarmonics + 2 + j) *= breathinessUnvoiced;
		}
	}
}

void applyStrength()
{

}

void applyBrightness()
{

}

void applyGrowl()
{

}

void applyCoarseness()
{
	//random voiced phase modulastion
}
/*Flags:
-pitch stability
-loop overlap
-loop offset
-fade in/out (i/o/io/x)
-steadiness
-breathiness
-formant shift
-intonation strength(?) (shift uv spectrum with v harms)
-subharmonics(?)
-peak dampening
-strength
-brightness
-coarseness
-growl
*/