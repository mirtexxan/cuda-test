% create some dummy test data
n_filters = 3;
n_neurons = 6;
timestamps = 4;
m_list = cell(1,n_filters);
for i = 1:n_filters
    m_list{i} = single(randn(n_neurons,timestamps));
    l_list{i} = randi([1,max(n_neurons, timestamps)-1]);
end

% import the python module
% NOTE: the py file MUST be in the same directory as the .m file
mod = py.importlib.import_module('run');
% Call the "run" method that execute the crosscorrelation between a list of matrices (m_list)
%        at a given lag, which can be different for each matrix (l_list)
% Note that the lenght of the two lists must be the same, of course.
ret = mod.run(m_list, l_list);

% Convert the return values from python data to matlab data.
% NOTE that no actual memory transfer should take place here... (need to check to be sure)
MAX_CORRELATIONS = cell(1,n_filters);
MAX_LAG = cell(1,n_filters);
for i = 1:n_filters
    tmp = single(ret{i});
    MAX_CORRELATIONS{i} = single(squeeze(tmp(1,:,:)));
    MAX_LAG{i} = single(squeeze(tmp(2,:,:)));
end

% let's print the return values for the first filter (i=1), as an example

% MAX_CORRELATIONS is a list containing (for each filter i) the maximum of the absolute value of the crosscorrelation.
% Note that MAX_CORRELATIONS{i} is a symmetric matrix, and the diagonal contains the autocorrelations
display(MAX_CORRELATIONS{1})
% MAX LAG is a list containing (for each filter i) the value of the LAG for which the abs of the crosscorrelation is the maximum.
%note that MAX_LAG{i} MUST be both antysimmetric and diag(max_lag) = [0...0]
display(MAX_LAG{1})
