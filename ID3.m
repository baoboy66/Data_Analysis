% Bao Do; David Beck; Cory McPhillips
%{
   Note:
    Run this probgram to get the graph for all 4 different bin values
   Examples results:
                Min     Max       Average
   5 bins => 95.4545  100.0000   98.3837
   10 bins => 95.4545  100.0000   97.9711
   15 bins => 92.3077  100.0000   97.7217
   20 bins => 95.4545  100.0000   98.3964
%}
function[] = ID3()
    load iris.dat; 
    plotData = [];
    for bin = 5:5:20;
        accuracy = [];
        for loop = 1:10
            data = modifyData(iris,bin);
            [trainSet, testSet] = splitMatrix(data);
            index = 1:length(testSet);
            prediction = getTrainingSet(trainSet, testSet,index);
            expectedSetosa = testSet(find(prediction == 1),5);   
            expectedValue = sum(expectedSetosa == 1);
            actualValue = sum(testSet(:,5) == 1);
            accuracy = [accuracy, expectedValue/actualValue];
        end
        minA = min(accuracy);
        maxA = max(accuracy);
        averageA = sum(accuracy)/length(accuracy);
        binResult = [minA, maxA, averageA] * 100;
        plotData = [plotData; binResult];
    end    
    disp(plotData);
    %plot data in bar graph
    bins = 5:5:20;   
    bar(bins, plotData)
    set(gca,'XTickLabel',{'5', '10', '15', '20'});
    xlabel('# of bins');
    ylabel('Accuracy in Percentage');
    title('ID3 Graph');
    legend({'Min','Max','Average'});   
end

%ID3 program: this is the recursive step
%param: data => training dataset; testData => test dataset, index of
%current testData
%return: predicted value for testData
function tData = getTrainingSet(data, testData, classIndex)
    tData = zeros(length(testData),1);
    uv = unique(data(:,length(data(1,:))));
    % define base case
    if length(uv) == 1 || length(data(1,:)) == 1
        tData(classIndex) = uv(1,1);
    else      
        classCol = length(data(1,:));
        class = data(:,classCol);
        gainInfo = [];
        for i = 1:length(data(1,:))-1
            gainInfo = [gainInfo, getGainInfo(data(:,i),class)];
        end
        % find the best attribute to split at
        splitPoint = find(gainInfo == max(gainInfo));
        splitPoint = splitPoint(1,1);       
        partitions = unique(data(:,splitPoint));
        for k = 1:length(partitions)
         newdata = data(data(:,splitPoint) == partitions(k,1), :);
         newdata(:,splitPoint) = [];
         subsetIndex = find(testData(classIndex,splitPoint) == partitions(k,1));
         newClassIndex = classIndex(subsetIndex);
         if length(newClassIndex) > 0
             cl = getTrainingSet(newdata, testData, newClassIndex);
             tData = tData + cl;
         end
        end
        
    end
end

%Randonize and plit the matrix into 2 smaller matrices
%return: 2 randonized matrices
function [A, B] = splitMatrix(data)
    n = length(data(:,1));
    rand = randperm(n);
    A = data(rand(1,1:n/2),:);
    B = data(rand(1,n/2+1:n),:);
end

% arguments: col - column to calculate frequency
% bin divide the column into equal bin
% return entropy value for that column
function entropy = getEntropy(col, classCol)
    entropy = [];
    uv = unique(col);
    for i = 1:length(uv)
        k = col(col == uv(i,1));
        class = classCol(col == uv(i,1));
        uvOfClass = unique(class);
        pAtNode = hist(class,length(uvOfClass));
        prob = pAtNode / sum(pAtNode);
        entropy = [entropy, -sum(prob .* log2(prob))];
    end
end

%Calculate the total entropy of the matrix with class column
%return the totalEntropy
function totalEntropy = getTotalEntropy(class)
    uv = unique(class);
    n = hist(class, length(uv));
    prob = n/sum(n);
    totalEntropy = -sum(prob .* log2(prob));
end

% modify the data by classify each data by their bins value
%Param: data => iris data set; bin => number of partitions of equal width
%return: Matrix with equal dimension with value of the bin it's long to
function result = modifyData(data, bin)
    result = [];
    for k = 1:length(data(1,:)) - 1
        col = data(:,k);
        diff = ((max(col)-min(col))/bin);
        a = 0:bin;
        % x is the set of splitting points
        x = min(col) + diff * a;  
        for i = 1:length(x) - 1
            data(col >= x(1,i) & col <= x(1,i+1), k) = i;
        end
        result = data;
    end
end

%Calculating the gainInfo of an attribute
%param: col => attribute column; classCol => class Column
%return: an array of gainInfo for each bin
function gain = getGainInfo(col, classCol)
    totalEnt = getTotalEntropy(classCol);
    uv = unique(col);
    n = hist(col,length(uv));
    entropies = getEntropy(col,classCol);
    gain = totalEnt - sum((n/sum(n)) .* entropies);
end