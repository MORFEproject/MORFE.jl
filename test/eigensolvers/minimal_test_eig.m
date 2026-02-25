%% import sparse matrices from csv file

tab=readtable("M.csv");
vals=tab.Variables;
% first line of vals is:
% "size", # of rows, # of cols
% other lines are:
% index rows, index columns, entry value
n_rows = vals(1,2);
n_cols = vals(1,3);
i_rows = vals(2:end,1);
j_cols = vals(2:end,2);
s_entry = vals(2:end,3);
M=sparse(i_rows, j_cols, s_entry, n_rows, n_cols);


tab=readtable("K.csv");
vals=tab.Variables;
% first line of vals is:
% "size", # of rows, # of cols
% other lines are:
% index rows, index columns, entry value
n_rows = vals(1,2);
n_cols = vals(1,3);
i_rows = vals(2:end,1);
j_cols = vals(2:end,2);
s_entry = vals(2:end,3);
K=sparse(i_rows, j_cols, s_entry, n_rows, n_cols);


%% compute eigenvalues

sigma = 10+1i*0;
n = 100;

[V,D,W] = eigs(-K,M,n,sigma,'Tolerance',1E-16);

D = diag(D);

scatter(real(D),imag(D),40,real(D),'filled')