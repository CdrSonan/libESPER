#include <stdio.h>

#define nTests 4

typedef struct
{
    float* indata;
    float* outdata;
    int inlength;
    int outlength;
}
testData;

typedef struct
{
    FILE* input;
    FILE* output;
    int inputOffsets[nTests];
    int outputOffsets[nTests];
    int index;
}
testConfig;

testData readWithConfig(testConfig config)
{
    int start;
    if (config.index == 0)
    {
        start = 0;
    }
    else
    {
        start = config.inputOffsets[config.index - 1];
    }
    int end = config.inputOffsets[config.index];
    int inlength = end - start;
    float* indata = (float*)malloc((end - start) * sizeof(float));
    fread(indata, sizeof(float), end - start, config.input);
    if (config.index == 0)
    {
        start = 0;
    }
    else
    {
        start = config.inputOffsets[config.index - 1];
    }
    end = config.inputOffsets[config.index];
    float* outdata = (float*)malloc((end - start) * sizeof(float));
    fread(outdata, sizeof(float), end - start, config.output);
    testData returnVal;
    returnVal.indata = indata;
    returnVal.outdata = outdata;
    returnVal.inlength = inlength;
    returnVal.outlength = end - start;
    return returnVal;
}

int testPitchCalcFallback(testConfig config)
{
    testData data = readWithConfig(config);
    return 0;
}

int testSpecCalc(testConfig config)
{
    testData data = readWithConfig(config);
    return 0;
}

int testResampleSpecharm(testConfig config)
{
    testData data = readWithConfig(config);
    return 0;
}

int testResamplePitch(testConfig config)
{
    testData data = readWithConfig(config);
    return 0;
}

int main(void)
{
	printf("Running tests...\n");
	return 0;
    FILE* input = fopen("test_data/test_input.bin", "rb");
    int inputOffsets[nTests];
    fread(inputOffsets, sizeof(int), nTests, input);
    FILE* output = fopen("test_data/test_output.bin", "rb");
    int outputOffsets[nTests];
    fread(outputOffsets, sizeof(int), nTests, output);
    testConfig config;
    config.input = input;
    config.output = output;
    for (int i = 0; i < nTests; i++) {
        config.inputOffsets[i] = inputOffsets[i];
        config.outputOffsets[i] = outputOffsets[i];
    }
    int returnCode = 0;
    config.index = 0;
    returnCode += testPitchCalcFallback(config);
    config.index = 1;
    returnCode += 2 * testSpecCalc(config);
    config.index = 2;
    returnCode += 4 * testResampleSpecharm(config);
    config.index = 3;
    returnCode += 8 * testResamplePitch(config);
    fclose(input);
    fclose(output);
    return returnCode;
}
