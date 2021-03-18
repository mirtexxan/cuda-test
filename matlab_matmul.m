rows = 63;
cols = 426;
tic
A = gpuArray(single(ones(rows,cols)));
B = gpuArray(single(ones(cols,rows)));
P = A*B;
toc

tic
mod = py.importlib.import_module('run');

C = single(mod.test_matmult(gather(A), gather(B)));
toc

isequal(P, C)
