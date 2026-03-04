%% ---------- helper: load sparse matrix from your CSV format ----------
function A = loadSparseCSV(fname)
    tab  = readtable(fname);          % assumes your file has the same layout you described
    vals = tab.Variables;

    n_rows = vals(1,2);
    n_cols = vals(1,3);

    i_rows = vals(2:end,1);
    j_cols = vals(2:end,2);
    s_entry = vals(2:end,3);

    A = sparse(i_rows, j_cols, s_entry, n_rows, n_cols);
end

%% ---------- helper: quick singularity / conditioning diagnostics ----------
function info = checkSingularitySparse(A, name)
    [m,n] = size(A);
    info.name = name;
    info.size = [m n];
    info.nnz  = nnz(A);
    info.square = (m == n);

    fprintf('%s: %d x %d, nnz=%d\n', name, m, n, info.nnz);

    if ~info.square
        fprintf('  %s is NOT square -> generalized eigs still may work depending on B, but singularity check is different.\n', name);
        info.is_singular_likely = NaN;
        return;
    end

    % Structural rank (fast; detects guaranteed singular patterns, but can miss numeric singularity)
    info.sprank = sprank(A);
    fprintf('  sprank(%s) = %d (out of %d)\n', name, info.sprank, n);

    % Condition estimate (sparse; Inf usually indicates singular or breakdown)
    try
        info.condest = condest(A);
    catch
        info.condest = NaN;
    end
    fprintf('  condest(%s) ≈ %g\n', name, info.condest);

    % Heuristic flags
    % - sprank < n  => structurally rank deficient (definitely singular)
    % - condest = Inf or huge => very ill-conditioned / singular-like
    hugeCond = 1e16;

    info.is_singular_likely = false;
    if info.sprank < n
        info.is_singular_likely = true;
        fprintf('  WARNING: %s appears structurally singular (sprank < n).\n', name);
    end
    if isinf(info.condest) || (~isnan(info.condest) && info.condest > hugeCond)
        fprintf('  WARNING: %s appears very ill-conditioned (condest large/Inf).\n', name);
    end

end

%% ---------- main script ----------
% load
M = loadSparseCSV("M.csv");
K = loadSparseCSV("K.csv");

% print + checks
infoM = checkSingularitySparse(M, "M");
infoK = checkSingularitySparse(K, "K");


%% ---------- eigs: shift-invert around sigma ----------
A = -K;
B = M;

sigma = 1 + 1i*0;
nEV   = 100;

opts = struct();
opts.tol   = 1e-16;

tic
[V,D,W] = eigs(A, B, nEV, sigma, opts);
lam = diag(D);
toc

% sort indices by largest real part
[~,idx] = sort(real(lam), 'descend');

% reorder eigenvalues + right eigenvectors
lam = lam(idx);
V   = V(:,idx);

%% ---------- plotting ----------

figure('Color','w');
scatter(real(lam), imag(lam), 40, real(lam), 'filled', ...
        'MarkerFaceAlpha',0.85, 'MarkerEdgeAlpha',0.85);
grid on; box on;
xline(0,'k-'); yline(0,'k-');
xlabel('Real(\lambda)'); ylabel('Imag(\lambda)');
title(sprintf('eigs(A,B): sorted by largest Real(\\lambda), sigma=%g', sigma));
cb = colorbar; cb.Label.String = 'Real(\lambda)';

% % %% ---------- eigs: largestreal ----------
% A = -K;
% B = M;
% 
% sigma = 10 + 1i*0;
% nEV   = 20;
% 
% opts = struct();
% opts.tol   = 1e-16;
% 
% [V,D,W] = eigs(A, B, nEV, 'largestreal', opts);
% lam = diag(D);

