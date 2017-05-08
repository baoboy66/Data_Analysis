function [w, iterations, e]=GradientDescent(Data, Target, eta, error, epochs)
    [rd, cd]=size(Data);
    [rt, ct]=size(Target);
    if rt ~= rt
        error('num data points not equal to num target');
    else
     w=rand(1,cd+1);
     iterations=0;
    e=error;
    while e >= error &&  iterations <= epochs
     iterations=iterations+1;
     wrong=0;
     for i=1:rd,
         out(i) = sum(w .* [Data(i,:) + Data(i,:).^2,1]); 
         deltaw=eta*(Target(i)-out(i))*[Data(i,:),1];
         w=w+deltaw;
         err(i)=(Target(i)- out(i))^2;
         if err(i)>0
             wrong=wrong+1;
         end
     end  
    % error for delta rule
    e=sum(err)/rd;
    end
end