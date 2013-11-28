% Compiles mexfiles on Linux.
% Only tested on the Cogito.ML cluster.

INCLUDE_PATHS   = {'/home/qho/boost_1_43_0/'};
LIBRARIES       = {'pthread'};
TARGETS         = {'mmsb_gs_core.cpp',
                   'mmsb_generative_core.cpp',
                   'mmsb_lls_core.cpp'};

compile_string  = 'mex -D_MEX ';
for include_path = 1:length(INCLUDE_PATHS)
    compile_string = [compile_string '-I' INCLUDE_PATHS{include_path} ' '];
end
for library = 1:length(LIBRARIES)
    compile_string = [compile_string '-l' LIBRARIES{library} ' '];
end
for target = 1:length(TARGETS)
    eval([compile_string TARGETS{target}]);
end

clear LIBRARIES TARGETS compile_string library target;
