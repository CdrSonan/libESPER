#!/bin/sh
echo "Running Octave nfsft tests..."
"" --eval "try; addpath('/home/sonan/repos/libESPER/nfft/matlab/tests','/home/sonan/repos/libESPER/nfft/matlab/nfsft'); nfsftUnitTestsRunAndExit; catch; disp('Error running nfsftUnitTestsRunAndExit'); end; exit(1);"
