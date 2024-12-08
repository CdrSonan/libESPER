#!/bin/sh
echo "Running MATLAB nfsft tests..."
"/matlab" -wait -nodesktop -nosplash -r "try; diary('check_nfsft_matlab.output'); addpath('/home/sonan/repos/libESPER/nfft/matlab/tests','/home/sonan/repos/libESPER/nfft/matlab/nfsft'); nfsftUnitTestsRunAndExit; catch; disp('Error running nfsftUnitTestsRunAndExit'); end; exit(1);"
