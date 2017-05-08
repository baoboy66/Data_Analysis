function [ w ] = logistic(single, combine1, combine2, eps, epochs )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
load iris.dat;
meas = iris(:,1:4);
[r, c]=size(meas);


% Extend meas by 1 to account for the bias
col1 = ones(r,1);
emeas=[col1 meas];


%Assign numerical labels
class=zeros(150,1);
class(1:50)=1;  %setosa
class(51:100)=2;  %versicolor
class(101:150)=3; %virginica

%Transform numerical lables into 0 / 1 labels
newclass(class==single) = 0; %setosa becomes class 0%


% the rest are in class 1
newclass(class == combine1) = 1;
newclass(class == combine2) = 1;

p = 0.1; % extent of training sets

randindex=randperm(r);
N = round(p*r)
train = emeas(randindex(1:N),:);
trainlabels = newclass(randindex(1:N));
test = emeas(randindex(N+1:r),:);
testlabels = newclass(randindex(N+1:r));

%Algorithm 8.2 without test for convergence
% initialize w
w=zeros(c+1,1);%only do this the first time
iterations = 0;
ybar=mean(trainlabels);
w(1)=log(ybar/(1-ybar));
s=zeros(1,N);
z=zeros(N,1);
while (iterations < epochs)
    iterations = iterations + 1;
     for i=1:N,
       % learning rate
       ni = w(1) + w.' .* train(i,:);
       temp = sum(ni)/length(ni);
       ni = zeros(N,1);
       ni(:,1)=temp;
       y = 1 ./ (1 + exp(-ni));
       si = y .* (1 - y) + eps;
       zi = temp + (newclass(i) - y)/si;

     end  % for i=1:rd 
       S = diag(si);
       w = pinv(train.' * S * train) * train.' * S * z;
end



%HERE COMES ALG. 8.2


%Test
ltest=length(testlabels);


% Compute the output for each test data using w
for i=1:ltest,
out(i)=test(i,:)*w;
end

% Transform output in 0 1 labels
out1=out;
out1(out<0)=0;
out1(out>0)=1;

% compute accuracy
accuracy = 1 - sum(abs(testlabels - out1))/ltest

% plot confusion matrix
plotconfusion(testlabels, out1);

%plot ROC curves
plotroc(testlabels, out1);
    

end

