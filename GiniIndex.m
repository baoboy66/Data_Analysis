function [  ] = HW(  )
    data = xlsread('glassdataB.xls');
    col = data(:,2);
    minCol4 = min(col);
    maxCol4 = max(col);
    splittingPoint = 4;
    intervalDiff = (maxCol4 - minCol4)/splittingPoint;
    ginis = [];
    gain = [];
    gainRatio = [];
    info = [];
    col11 = data(:,11);
    overAllE = [getEntropy(col11)];
    a = 0;
    n = 214;
    for i = 1:splittingPoint-1
        class = data(col <= minCol4 + intervalDiff * i, 11);
        ni = length(class);
        if ni == 0
            ginis = [ginis, 0]       
        else     
            gini = getGini(class);
            gini2 = getGini(col11(length(class):214));
            ginis = [ginis, (ni/n) * gini + ((n-ni)/n) * gini2]
        end
    end
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


