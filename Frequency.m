function [  ] = HW(  )
    data = xlsread('glassdataB.xls');
    col1 = data(:,1);
    intervalDiff = 214/4;
    ginis = [];
    gain = [];
    gainRatio = [];
    info = [];
    col11 = data(:,11);
    overAllE = [getEntropy(col11)]
    a = 0;
    n = 214;
    for i = 1:3
        class = data(col1 <= intervalDiff * i, 11);
        ni = length(class);
        gini = getGini(class);
        gini2 = getGini(col11(length(class):214));
        ginis = [ginis, (ni/n) * gini + ((n-ni)/n) * gini2]
        entropy = getEntropy(class);
        entropy2 = getEntropy(col11(length(class):214));
        gain = [gain, overAllE - ((ni/n) * entropy) - ((n - ni)/n * entropy2)]
        info = [info, splitInfo(ni, n)];
    end
        GainRatio = gain ./ info
end

function gini = getGini(col)
    uv = unique(col);
    occ = histc(col, uv);
    occ = occ / sum(occ(:));
    i = find(occ);
    gini = 1 - sum(occ(i) .^2);
end

function entropy = getEntropy(col)
    uv = unique(col);
    occ = histc(col, uv);
    occ = occ / sum(occ(:));
    i = find(occ);
    entropy = -sum(occ(i) .* log2(occ(i)));
end

function inf = splitInfo(ni, n)
    inf = -((ni/n) * log2(ni/n));
    inf = inf - (((n-ni)/n) * log2((n-ni)/n));
end
