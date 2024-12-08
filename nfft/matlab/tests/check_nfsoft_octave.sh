#!/bin/sh
echo "Running Octave nfsoft tests..."
"" --eval "try; addpath('/home/sonan/repos/libESPER/nfft/matlab/tests','/home/sonan/repos/libESPER/nfft/matlab/nfsft','/home/sonan/repos/libESPER/nfft/matlab/nfsoft'); perform_exhaustive_tests_flag=0; nfsoftUnitTestsRunAndExit; catch; disp('Error running nfsoftUnitTestsRunAndExit'); end; exit(1);"
