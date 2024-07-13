//Copyright 2023 Johannes Klatt

//This file is part of libESPER.
//libESPER is free software: you can redistribute it and /or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.
//libESPER is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//You should have received a copy of the GNU General Public License along with Nova - Vox.If not, see <https://www.gnu.org/licenses/>.

#pragma once

#include "src/util.h"

float* lowRangeSmooth(cSample sample, float* signalsAbs, engineCfg config);

float* highRangeSmooth(cSample sample, float* signalsAbs, engineCfg config);

void finalizeSpectra(cSample sample, float* lowSpectra, float* highSpectra, engineCfg config);

void separateVoicedUnvoiced(cSample sample, engineCfg config);

void finalizeSample(cSample sample, engineCfg config);
