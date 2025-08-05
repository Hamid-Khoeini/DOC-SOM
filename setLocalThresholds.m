function localThreshold = setLocalThresholds(pCenters, gammaValIter, net)
    % Inputs:
    % - pCenters: a cell array containing local density estimates at each neuron center 
    %             (each cell is of size [numFeatures x numCenters])
    % - gammaValIter: scaling factor (e.g., 0.05)
    % - net: structure containing:
    %        - net.W: neuron weights (d x N)
    %        - net.grd: SOM grid coordinates (2 x N)

    numCenters = size(pCenters, 2);             % Number of neurons
    numFeatures = size(pCenters{1}, 1);         % Number of input dimensions
    localThreshold = zeros(numFeatures, numCenters);  % Threshold matrix

    neuronPos = net.W';                          % Neuron positions [N x d]
    grid      = net.grd;                         % SOM grid positions [2 x N]
    adjMat    = linkdist(grid) == 1;             % Adjacency matrix based on neighborhood [N x N]

    for j = 1:numCenters
        neighborsIdx = find(adjMat(j, :));       % Indices of neighboring neurons

        mu_j = neuronPos(j, :);                  % Position vector of neuron j [1 x d]

        if isempty(neighborsIdx)
            thetaPerDim = zeros(numFeatures, 1); % If no neighbors, angle is zero for all dimensions
        else
            mu_neighbors = mean(neuronPos(neighborsIdx, :), 1);  % Mean position of neighbors [1 x d]

            thetaPerDim = zeros(numFeatures, 1);
            for i = 1:numFeatures
                u = mu_j(i);
                v = mu_neighbors(i);
                dotProd = u * v;
                norms = norm(u) * norm(v);
                if norms == 0
                    thetaPerDim(i) = 0;          % Avoid division by zero
                else
                    cosTheta = dotProd / norms;
                    cosTheta = max(min(cosTheta, 1), -1);  % Clamp to [-1, 1] for numerical stability
                    thetaPerDim(i) = acos(cosTheta);       % Compute angle in radians
                end
            end
        end

        % Scale thresholds by dimension-wise angle adjustment
        thetaScale = 1 + thetaPerDim / pi;                    % Normalize by π
        localThreshold(:, j) = gammaValIter .* pCenters{j}(:, j) .* thetaScale;
    end
end

