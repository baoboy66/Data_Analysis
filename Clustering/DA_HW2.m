function [ output_args ] = DA_HW2(  )
    %load data
    data = xlsread('data_banknote_authentication.xlsx');
    % calculate covariance matrix of the data
    cm = cov(data);
    %getScatterPlot(1,2,data)
    %getScatterPlot(2,4,data)
    %getScatterPlot(3,5,data)
    %getScatterPlot(2,5,data)
    [training, test] = splitMatrix(data, 1000);
    
    size = [5, 25, 50];
    for i = 1:3
        xTraining = training(:,1:4);
        yTraining = training(:,5);
        tree = fitctree(xTraining, yTraining,'MinLeafSize', size(i));
        view(tree, 'mode','graph');
        childNodes = tree.Children;
        leafNodes = sum(childNodes(:,1) == 0);
        fprintf('MinLeafSize: %d \n  Number of leaf nodes: %d \n', size(i), leafNodes);
    
        depth = treeDepth(tree, length(tree.Parent));
        depthCount = longestPathCount(tree, depth);
        lastNode = length(tree.Parent);
        population = tree.ClassCount(lastNode);
        purity = getPurity(tree,lastNode);
        fprintf('Longest path or depth: %d. \n', depth);
        fprintf('Number of longest paths: %d \n', depthCount);
        fprintf('Population: %d \n', population);
        fprintf('Purity: %0.3f \n', purity);
        
        prediction = predict(tree, test(:,1:4));
        conMatrix = confusionmat(test(:,5), prediction);
        fprintf('Confusion Matrix \n');
        disp(conMatrix);
        a = conMatrix(1,1);
        b = conMatrix(1,2);
        c = conMatrix(2,1);
        d = conMatrix(2,2);
        
        accuracy = plus(a,d)/(a + b + c + d) * 100;
        recall = a/plus(a,b) * 100;
        precision = a/plus(a,c) * 100;
        
        fprintf('Accuracy: %0.2f%% \nRecall: %0.2f%% \nPrecision: %0.2f%% \n \n',accuracy, recall, precision);
    end

    %view(tree);
    %tree = fitctree(training(:,1:4),training(:,5),'MinLeafSize', 25);
    %view(tree, 'mode','graph');
    %view(tree)
    %tree = fitctree(training(:,1:4),training(:,5),'MinLeafSize', 50);
    %view(tree, 'mode','graph');
    %view(tree);
end

%calculate purity using the entropy formula
%take in a decision tree, output the purity of the longest path which is
%the last node in the array
function purity = getPurity(tree, index)
    prob = [tree.ClassProbability(index,1), tree.ClassProbability(index,2)];
    purity = 0;
    if(length(find(prob == 0)) == 0)
        purity = -sum(prob .* log2(prob));
    end
end

function depth = treeDepth(tree, index)
    parent = tree.Parent;
    depth = 0;
    node = parent(index);
    while node~=0
        depth = depth + 1;
        node = parent(node);
    end
end

function count = longestPathCount(tree, depth)
    parent = tree.Parent;
    index = [];
    count = 0;
    for i = 1:length(parent)
        n = treeDepth(tree, i);
        if n == depth
            count = count + 1;
            index = [index,i];
        end
    end
end

function getScatterPlot(att1, att2, data)
    clear plot;
    D10 = data(data(:,5) == 0,:)
    D11 = data(data(:,5) == 1,:);
    scatter(D10(:,att1),D10(:,att2),'k');
    hold on
    scatter(D11(:,att1),D11(:,att2),'r');
    hold off
    str = sprintf('Att %d vs Att %d', att1,att2);
    title(str);
    legend('C0', 'C1');
    lblY = sprintf('Attribute %d',att2);
    lblX = sprintf('Attribute %d',att1);
    ylabel(lblY);
    xlabel(lblX);
end

%Randonize and plit the matrix into 2 smaller matrices with first matrix of
%length ALength
%return: 2 randonized matrices
function [A, B] = splitMatrix(data, ALength)
    n = length(data(:,1));
    rand = randperm(n);
    A = data(rand(1,1:ALength),:);
    B = data(rand(1,ALength+1:n),:);
end