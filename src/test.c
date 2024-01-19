#include <stdio.h>

const uint nTests = 5;

int main(void) {
  FILE* input = fopen("test_input.bin", rb);
  uint[nTests] inputOffsets;
  fread(inputOffsets, sizeof(uint), nTests, input);
  FILE* output = fopen("test_output.bin", rb);
  uint[nTests] outputOffsets;
  fread(outputOffsets, sizeof(uint), nTests, output);
  returnCode = 0;
  
  fclose(input);
  fclose(output);
  return returnCode;
}

float* readWithIndex(FILE* file, uint[nTests] offsets, uint index) {
  if (index == 0) {
    uint start = 0;
  } else {
    uint start = offsets[index - 1];
  }
  uint end = offsets[index];
  float* content = malloc((end - start) * sizeof(float));
  fread(content, sizeof(float), end - start, file);
}
