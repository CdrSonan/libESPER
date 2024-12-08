#!/bin/sh
echo "Running Octave nfft tests..."
"" --eval "try; addpath('/home/sonan/repos/libESPER/nfft/matlab/tests','/home/sonan/repos/libESPER/nfft/matlab/nfft'); nfftUnitTestsRunAndExit; catch; disp('Error running nfftUnitTestsRunAndExit'); end; exit(1);"
