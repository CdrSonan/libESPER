#!/bin/sh
echo "Running MATLAB nfsoft tests..."
"/matlab" -wait -nodesktop -nosplash -r "try; diary('check_nfsoft_matlab.output'); addpath('/home/sonan/repos/libESPER/nfft/matlab/tests','/home/sonan/repos/libESPER/nfft/matlab/nfsft','/home/sonan/repos/libESPER/nfft/matlab/nfsoft'); perform_exhaustive_tests_flag=0; nfsoftUnitTestsRunAndExit; catch; disp('Error running nfsoftUnitTestsRunAndExit'); end; exit(1);"
