% ========================================================
% MASTER ANALYSIS SCRIPT
% Covers Sections 3.1, 3.3, 4.1, 4.2 of the Project Report
% ========================================================

clear; close all; clc;
addpath(genpath( ...
    '/Users/andrei/Library/CloudStorage/OneDrive-Personal/Documents/University/Semester 1/DSML/DSML-Project---Team-2/Project/Team 2-code_PLRNNs/code_PLRNNreg_BOLD_SSM' ...
));
rehash;
clear classes;

disp('--- Sanity check: PLRNN_BOLD on path? ---');
which PLRNN_BOLD -all
disp(['exist(PLRNN_BOLD,''class'') = ', num2str(exist('PLRNN_BOLD','class'))]);
disp('----------------------------------------');

% ------------
% Directories
% ------------
thisFileDir = fileparts(mfilename('fullpath'));
patData     = fullfile(thisFileDir, 'data');
patOut      = fullfile(patData, 'output');
patFigs     = fullfile(patOut, 'final_analysis_figures');
if ~isfolder(patFigs), mkdir(patFigs); end

% ----------
% Settings
% ----------
nSubjects = 10;
M = 20;

% ---------
% ---------
LL_baseline     = nan(nSubjects, 1);
LL_constrained  = nan(nSubjects, 1);
W_stack         = zeros(M, M, nSubjects);
All_Eigenvalues = [];

% For E/I indices
idx_E = [];
idx_I = [];

fprintf('Loading data from: %s\n', patOut);

% =========================================================================
% LOADING AND COLLECTION OF STATS PER SUBJECT
% =========================================================================
for k = 1:nSubjects

    fileBase  = fullfile(patOut, sprintf('outputfile%d.mat', k));
    fileFinal = fullfile(patOut, sprintf('outputfile%d_w_sparsity_modularity_ei.mat', k));

    % 2) Load Baseline
    if isfile(fileBase)
        dBase = load(fileBase);
        if isfield(dBase, 'LL') && ~isempty(dBase.LL)
            LL_baseline(k) = dBase.LL(end);
        end
    else
        fprintf('  Subject %d: Baseline file missing.\n', k);
    end

    % 3) Load Final
    if ~isfile(fileFinal)
        fprintf('  Subject %d: Final file missing.\n', k);
        continue;
    end

    d = load(fileFinal);

    % Extract LL
    if isfield(d, 'LL') && ~isempty(d.LL)
        LL_constrained(k) = d.LL(end);
    end

    % Extract Network Object (net_sparse preferred)
    if isfield(d, 'net_sparse')
        curr_net = d.net_sparse;
    elseif isfield(d, 'net')
        curr_net = d.net;
    else
        warning('Subject %d: No net/net_sparse found. Skipping.', k);
        continue;
    end

    if isobject(curr_net)
        if ~(isprop(curr_net,'W') && isprop(curr_net,'A'))
            warning('Subject %d: curr_net missing W/A properties. Skipping.', k);
            continue;
        end
    elseif isstruct(curr_net)
        if ~(isfield(curr_net,'W') && isfield(curr_net,'A'))
            warning('Subject %d: curr_net missing W/A fields. Skipping.', k);
            continue;
        end
    else
        warning('Subject %d: curr_net is %s. Skipping.', k, class(curr_net));
        continue;
    end

    % Extract matrices
    W = curr_net.W;
    A = curr_net.A;

    % Store W
    if ~isequal(size(W), [M M])
        warning('Subject %d: W is size %s (expected %dx%d). Skipping W storage.', ...
            k, mat2str(size(W)), M, M);
    else
        W_stack(:,:,k) = W;
    end

    % ---------------------------------------------------------------------
    % E/I index extraction (works for objects or structs)
    % ---------------------------------------------------------------------
    if isempty(idx_E) || isempty(idx_I)

        EI_sign = [];

        % Case 1: object has a direct EI_sign property
        if isobject(curr_net) && isprop(curr_net,'EI_sign')
            EI_sign = curr_net.EI_sign;
        end

        % Case 2: object has a "reg" property that contains EI_sign
        if isempty(EI_sign) && isobject(curr_net) && isprop(curr_net,'reg')
            try
                if isstruct(curr_net.reg) && isfield(curr_net.reg,'EI_sign')
                    EI_sign = curr_net.reg.EI_sign;
                elseif isobject(curr_net.reg) && isprop(curr_net.reg,'EI_sign')
                    EI_sign = curr_net.reg.EI_sign;
                end
            catch
            end
        end

        % Case 3: EI_sign exists as a top-level loaded variable in the file
        if isempty(EI_sign) && isfield(d,'EI_sign')
            EI_sign = d.EI_sign;
        end

        if ~isempty(EI_sign)
            idx_E = find(EI_sign > 0);
            idx_I = find(EI_sign < 0);

            fprintf('Found EI_sign on subject %d: |E|=%d, |I|=%d\n', k, numel(idx_E), numel(idx_I));
        else
            if k == 1
                warning('No EI_sign found in curr_net/file. Figure 4.2 may be skipped.');
            end
        end
    end

    % ---------------------------------------------------------------------
    % Collect Eigenvalues for Stability (Proxy Jacobian = A + W)
    % ---------------------------------------------------------------------
    if isequal(size(A), [M M]) && isequal(size(W), [M M])
        J = A + W;
        All_Eigenvalues = [All_Eigenvalues; eig(J)]; %#ok<AGROW>
    end

end

% =========================================================================
% OUTLIER REMOVAL / VALID SUBJECT FILTER
% =========================================================================
valid_mask = ~isnan(LL_baseline) & ~isnan(LL_constrained);
valid_mask = valid_mask & (LL_constrained > -1e6) & (LL_baseline > -1e6);

nRemoved = nSubjects - sum(valid_mask);
if nRemoved > 0
    fprintf('\n!!! REMOVED %d OUTLIER SUBJECTS (Crashed/Failed) !!!\n', nRemoved);
    fprintf('analyzing %d valid subjects.\n', sum(valid_mask));
end
fprintf('Valid subjects:   %s\n', mat2str(find(valid_mask)));
fprintf('Removed subjects: %s\n', mat2str(find(~valid_mask)));

LL_base_clean = LL_baseline(valid_mask);
LL_cons_clean = LL_constrained(valid_mask);
W_stack_clean = W_stack(:,:,valid_mask);

% =========================================================================
% FIGURE 3.1: Data Fidelity Comparison
% =========================================================================
if ~isempty(LL_base_clean)
    figure('Color','w', 'Position', [100 100 650 420]);

    bar_data = [mean(LL_base_clean), mean(LL_cons_clean)];
    err_data = [std(LL_base_clean)/sqrt(length(LL_base_clean)), std(LL_cons_clean)/sqrt(length(LL_cons_clean))];

    b = bar(bar_data, 'FaceColor', 'flat');
    b.CData(1,:) = [0.6 0.6 0.6]; % Grey
    b.CData(2,:) = [0.2 0.7 0.3]; % Green
    hold on;
    errorbar(1:2, bar_data, err_data, 'k.', 'LineWidth', 2);

    ylabel('Log-Likelihood (Higher = Better Fit)');
    xticklabels({'Baseline', 'Sparse+Mod+E/I'});
    title('3.1 Data Fidelity Comparison (Valid Subjects)');
    grid on;
    saveas(gcf, fullfile(patFigs, '3.1_Data_Fidelity.png'));

    figure('Color','w', 'Position', [120 120 650 450]);
    plot([1 2], [LL_base_clean LL_cons_clean]', '-o', 'LineWidth', 1.5);
    xlim([0.75 2.25]); grid on;
    xticks([1 2]); xticklabels({'Baseline','Sparse+Mod+E/I'});
    ylabel('Log-Likelihood');
    title('3.1b Paired Subject-wise Log-Likelihood');
    saveas(gcf, fullfile(patFigs, '3.1b_Paired_LL.png'));
else
    warning('No valid subjects for Data Fidelity plot.');
end

% =========================================================================
% FIGURE 3.3: Dynamical Stability (Proxy)
% =========================================================================
if ~isempty(All_Eigenvalues)
    figure('Color','w', 'Position', [150 150 520 520]);
    theta = 0:0.01:2*pi;
    plot(cos(theta), sin(theta), 'k--', 'LineWidth', 1.5); hold on;
    plot(real(All_Eigenvalues), imag(All_Eigenvalues), '.', 'Color', [0.8 0.2 0.2], 'MarkerSize', 8);
    axis equal; grid on;
    xlabel('Real(\lambda)'); ylabel('Imag(\lambda)');
    title('3.3 Dynamical Stability');
    subtitle('Eigenvalues of Proxy Jacobian (A+W)');
    saveas(gcf, fullfile(patFigs, '3.3_Stability_Eigenvalues.png'));
else
    warning('No eigenvalues collected for stability plot.');
end

% =========================================================================
% FIGURE 4.1: Topological Interpretation (Group Average)
% =========================================================================
if size(W_stack_clean,3) > 0
    W_group_mean = mean(W_stack_clean, 3);

    figure('Color','w', 'Position', [200 200 740 620]);
    imagesc(W_group_mean);
    colormap(jet); colorbar;

    cmax = max(abs(W_group_mean(:)));
    if cmax == 0, cmax = 1; end
    clim([-cmax cmax]);
    axis square;
    title(sprintf('4.1 Group Average W (N=%d)', size(W_stack_clean,3)));
    xlabel('Source Region'); ylabel('Target Region');
    saveas(gcf, fullfile(patFigs, '4.1_Topology_Heatmap.png'));

    % =========================================================================
    % FIGURE 4.2: E/I Circuit Analysis
    % =========================================================================
    if ~isempty(idx_E) && ~isempty(idx_I)

        idx_E = idx_E(idx_E >= 1 & idx_E <= M);
        idx_I = idx_I(idx_I >= 1 & idx_I <= M);

        if ~isempty(idx_E) && ~isempty(idx_I)
            EE = abs(W_group_mean(idx_E, idx_E));
            EI = abs(W_group_mean(idx_I, idx_E));
            IE = abs(W_group_mean(idx_E, idx_I));
            II = abs(W_group_mean(idx_I, idx_I));

            % Exclude diagonal for within-type means
            mEE = sum(EE(:)) / max(1, numel(EE) - length(idx_E));
            mII = sum(II(:)) / max(1, numel(II) - length(idx_I));

            mEI = mean(EI(:));
            mIE = mean(IE(:));

            figure('Color','w', 'Position', [250 250 650 420]);
            bb = bar([mEE, mEI, mIE, mII]);
            bb.FaceColor = 'flat';
            bb.CData(1,:) = [0.9 0.8 0.2]; 
            bb.CData(2,:) = [0.2 0.8 0.2];
            bb.CData(3,:) = [0.8 0.2 0.2]; 
            bb.CData(4,:) = [0.2 0.2 0.8]; 

            xticklabels({'E \to E', 'E \to I', 'I \to E', 'I \to I'});
            ylabel('Mean Absolute Connection Strength');
            title('4.2 E/I Circuit Connectivity');
            grid on;
            saveas(gcf, fullfile(patFigs, '4.2_EI_Analysis.png'));
        else
            warning('idx_E/idx_I invalid after range check. Skipping Fig 4.2.');
        end

    else
        warning('idx_E/idx_I not found (no EI_sign available). Skipping Fig 4.2.');
    end
else
    warning('No valid subjects for topology plots.');
end

fprintf('Analysis Complete. Figures saved to: %s\n', patFigs);
