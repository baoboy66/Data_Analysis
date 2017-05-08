function [ output_args ] = Cluster( )
 data = xlsread('StudentData2.xlsx');
 DATA = data(1:50,2:5);

    K = 3:1:8;
    SSEList = zeros(1, length(K));
    AverageS = zeros(1, length(K));
    Clusterings = cell(0);

    for i = 1:length(K)
        [idx, c, sumD] = kmeans(DATA,K(i),'replicates',3);         
        SSEList(i) = sum(sumD);
        Clusterings(end+1) = {struct('idx',idx,'c',c,'sumD',sumD)};

        % plotting silhouette graphs
        subplot(2,3,i);
        [s h] = silhouette(DATA, idx, 'Euclidean');
        AverageS(i) = mean(s);
        ylabel('Clusters');
        xlabel('Silhouette Values');
        mytitle = sprintf('Sihouette for K = %d',K(i));
        title(mytitle);
    end
    
    %plotting SSE vs K 
    figure;
    plot(K,SSEList,'ko');
    ylabel('SSE');
    xlabel('Number of Clusters');
    mytitle = sprintf('SSE at K');
    title(mytitle);
    
    %Use silhouette coefficient to determine the best nubmer of cluster
    id = AverageS == max(AverageS);
    BestNumCluster = K(id);
    clustering1 = Clusterings{id}.idx;
    Centroid = Clusterings{id}.c;
    SSE = SSEList(id);
    
    %random data
    randData = 100 * rand(50,4);
    [randIdx, randC RandSumdD] = kmeans(randData, K(id));
    randSSE = sum(RandSumdD);
    population = [];
    for i = 1:BestNumCluster
        population = [population sum(randIdx == i)];
    end
    
    fprintf('Cluster1: \n Best number of clusters: %d \n', BestNumCluster);
    fprintf('Centroids: \n');
    disp(Centroid);
    fprintf('Random Data Centroids: \n');
    disp(randC);
    fprintf('SSE random data: %d \n', randSSE);
    fprintf('SSE from given data: %d \n', SSE);
    
    %plot clusering2 dendrogram
    D = pdist(DATA);
    cluster2 = linkage(D, 'single');
    dendrogram(cluster2);
    ylabel('Distance between clusters');
    xlabel('Clusters');
    title('Single Linkage');
    
    %plot clusering3 dendrogram
    cluster3 = linkage(D, 'complete');
    figure;
    dendrogram(cluster3);    
    ylabel('Distance between clusters');
    xlabel('Clusters');
    title('Complete Linkage');
        
    clustering2 = cluster(cluster2, 'maxclust', 4);
    clustering3 = cluster(cluster3, 'maxclust', 4);
    
    % getting the centroids for clustering2 and clustering3
    Centroids2 = zeros(4);
    Centroids3 = zeros(4);
    for i = 1:4
        currentCluster2 = DATA(clustering2 == i,:);   
        currentCluster3 = DATA(clustering3 == i,:); 
        Centroids2(i,:) = mean(currentCluster2);
        Centroids3(i,:) = mean(currentCluster3);
        fprintf('Clusters: \n');
        fprintf('Single Link: \n');
        disp(currentCluster2);
        fprintf('Complete Link: \n');
        disp(currentCluster3);
    end
    
    fprintf('Single Link Centroids: \n');
    disp(Centroids2);
    fprintf('Complete Link Centroids: \n');
    
    disp(Centroids3);
    % Rand Index for the camparison of Clustering2 and Clustering3  
    [randIdx a b c d] = RandIndex(clustering2, clustering3);
    fprintf('Cluster2 vs Cluster3 \n RandIndex : %d \na: %d \nb: %d \nc: %d \nd: %d\n', randIdx,a,b,c,d);
    % Rand Index for the camparison of Clustering1 and Clustering2 
    [randIdx a b c d] = RandIndex(clustering2, clustering1);
    fprintf('Cluster1 vs Cluster2 \n RandIndex : %d \na: %d \nb: %d \nc: %d \nd: %d\n', randIdx,a,b,c,d);
    % Rand Index for the camparison of Clustering2 and Clustering3 
    [randIdx a b c d] = RandIndex(clustering3, clustering1);
    fprintf('Cluster1 vs Cluster3 \n RandIndex : %d \na: %d \nb: %d \nc: %d \nd: %d\n', randIdx,a,b,c,d);
end

% calculat the rand index between two clusters
function [randidx, a,b,c,d] = RandIndex(c1, c2)
    a = 0; b = 0; c = 0; d = 0;
    len = length(c1);
    for i = 1:len
        for j = (i+1):len
            val1 = c1(i) == c1(j); 
            val2 = c2(i) == c2(j);
            if val1 && val2
                a = a + 1;
            elseif ~val1 && ~val2
                b = b + 1;
            elseif val1 && ~val2
                c = c + 1;
            else
                d = d + 1;
            end
        end
    end
    
    randidx = (a+b)/(a+b+c+d);
end
