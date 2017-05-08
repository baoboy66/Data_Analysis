function [w, iterations, e]=AdaptiveRate(Data, Target, eta, error, epochs, threshold, d, D)
%% Invoke as: [w, iterations, e] = AdaptiveRate(Data, Target, eta, error, epochs threshold, d, D)
%% implements the delta  rule;
%% Input:
%%  Data is a matrix N x P data points/vectors
%%  Target is vector N x 1 of target values (true output) corresponding to the data points
%%  eta: learning rate; 
%%  error : desired approximation error;
%%  epochs: threshold on the number of epochs (iterations through the whole
%% data set)
%%  d: value to lower learning rate 
%%  D: value to increase learning rate
%% Output:
%%  w is a vector of dimension P+1 x 1, where w_i is the weight for dimension i of a data point,
%%     for i=1:P, extended with weight w0 for the bias (input = 1)
%%  iterations = MIN{is the number of iterations taken to reach error threshold e, epochs}
%%  e: error threshold
    [rd, cd]=size(Data);
    [rt, ct]=size(Target);
    if rt ~= rt
        error('num data points not equal to num target');
    else
     w=rand(1,cd+1);
     iterations=0;
    e=error;
    
    % calculate initial output and error
    out(1) = sum(w .* [Data(1,:),1]);
    err(1)=(Target(1)- out(1))^2;
    
    while e >= error &&  iterations <= epochs
     iterations=iterations+1;
     wrong=0;
     for i=2:rd,
         out(i) = sum(w .* [Data(i,:),1]);  % delta rule 
         deltaw=eta*(Target(i)-out(i))*[Data(i,:),1];
         w=w+deltaw;
         err(i)=(Target(i)- out(i))^2;
         if err(i)>0
             wrong=wrong+1;
         end
         % discar weight vector and decrease learning rate
         if err(i) - err(i-1) > threshold
             w = w - deltaw;
             eta = eta * d;
         else
             eta = eta * D;
         end
     end  
    % error for delta rule
    e=sum(err)/rd;
    end
end