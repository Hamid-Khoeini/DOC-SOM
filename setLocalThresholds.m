function localThreshold = setLocalThresholds(pCenters, gammaValIter)

    numCenters = size(pCenters,2);
    localThreshold = zeros(size(pCenters{1},1), numCenters);

    for t = 1:numCenters
        localThreshold(:,t) = gammaValIter .* pCenters{t}(:,t);
    end

end
