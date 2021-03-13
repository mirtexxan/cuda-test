rows = 63;
cols = 426000;
A = single(ones(rows,cols));
B = single(ones(cols,rows));
P = A*B;


mod = py.importlib.import_module('run');
py.importlib.reload(mod);

C = single(mod.test_matmult(A, B));
isequal(P, C)