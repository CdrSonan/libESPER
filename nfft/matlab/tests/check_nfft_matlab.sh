#!/bin/sh
echo "Running MATLAB nfft tests..."
"/matlab" -wait -nodesktop -nosplash -r "try; diary('check_nfft_matlab.output'); addpath('/home/sonan/repos/libESPER/nfft/matlab/tests','/home/sonan/repos/libESPER/nfft/matlab/nfft'); nfftUnitTestsRunAndExit; catch; disp('Error running nfftUnitTestsRunAndExit'); end; exit(1);"
