clear all;
clc;
try
	Type = ver;
	% This part is for OCTAVE
	if(strcmp(Type(1).Name, 'Octave') == 1)
		mex trainLDM.c model_matlab.c LDM.cpp
		mex predictLDM.c model_matlab.c LDM.cpp
	% This part is for MATLAB
	% Add -largeArrayDims on 64-bit machines of MATLAB
    else
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims trainLDM.c model_matlab.c LDM.cpp
		mex CFLAGS="\$CFLAGS -std=c99" -largeArrayDims predictLDM.c model_matlab.c LDM.cpp
	end
catch
	fprintf('If make.m failes, please check README about detailed instructions.\n');
end
