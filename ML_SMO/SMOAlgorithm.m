function [ Sol, weightVec, bVal, alphs ] = SMOAlgorithm()
% Bao Do, David Beck, Cory Philips
% 20CS6037: Machine Learning
% Instructor: Anca Ralescu
% WARNING: when testing this, our runtime was taking between 15 and 20
% minutes to give us our results
    global colLength rowLength C eps alpha TARGET data storedErrors kern B upperLim;
    C = 0.5;
    eps = 0.001;
    B = 0;
    upperLim = 1;
    %IMPORTFILE Import numeric data from a text file as a matrix.
    data = importdata('heart.txt');
    [rowLength, colLength] = size(data);
    % set class valuue to -1 and 1
    data(data(:,colLength) ==2,colLength) = -1;
    %target is Y-values which is the class column
    TARGET = data(:, colLength);
    %alpha is set to zero
    alpha = zeros(1,rowLength);
    storedErrors = zeros(1, rowLength);
    numChanged = 0;
    examineAll = true;   
    kern = zeros(rowLength);
    for i = 1:rowLength
        for j = 1:rowLength
            kern(i,j) = kernel(i,j);
        end
    end
    while(numChanged > 0 || examineAll)
        numChanged = 0;
        if(examineAll)
            for i = 1:rowLength
                numChanged = numChanged + examineExample(i);
            end
        else
            boundChecked = find(storedErrors>C & storedErrors < upperLim-C);
            for i = boundChecked
                numChanged = numChanged + examineExample(i);
            end
        end
        if(examineAll)
            examineAll = false;
        elseif(numChanged == 0)
            examineAll = true;
        end
    end 
    Sol = 0;
    for i=1:rowLength
     if (SVMOutput(i) * TARGET(i) < 0)
         Sol = Sol + 1;
     end
    end
    bVal = B;
    Sol = Sol / rowLength * 100;
    alphs = alpha;
    weightVec = TARGET' .* alpha;
end

%kernel of x1,x2 is the dot product of x1.x2
function result = kernel(idx1, idx2)
    global data colLength
    point1 = data(idx1,1:colLength-1);
    point2 = data(idx2,1:colLength-1);
    result = sum(point1 .* point2);
end

function output = SVMOutput(idx)
    global TARGET alpha B kern
    output = sum(TARGET' .* alpha .* kern(idx,:)) - B;
end

function result = takeStep(idx1, idx2)
    global eps alpha TARGET storedErrors B C kern upperLim;
    result = false;
    
    if (idx1 == idx2)
        return;
    end
        
    % get alpha values
    alph1 = alpha(idx1);
    alph2 = alpha(idx2);
    y1 = TARGET(idx1);
    y2 = TARGET(idx2);
    %E1 = SVM output on point[i1] - y1
    if((alph1 < C || alph1 > upperLim-C))
        E1 = SVMOutput(idx1)-y1;
    else
        E1 = storedErrors(idx1);
    end
    if((alph2 < C || alph2 > upperLim-C))
        E2 = SVMOutput(idx2)-y2;
    else
        E2 = storedErrors(idx2);
    end
    s = y1*y2;

    %Computing L, H
    if(y1 == y2) 
        L = max(0,(alph2+alph1-upperLim));
        H = min(upperLim,alph2+alph1);
    else
        L = max(0, alph2-alph1);
        H = min(upperLim, upperLim+alph2-alph1);
    end  

    if L == H
        return
    end

    k11 = kern(idx1,idx1);
    k12 = kern(idx1,idx2);
    k22 = kern(idx2,idx2);
    eta = 2*k12 - k11 - k22;
    if (eta < 0)
        a2 = alph2 - y2*(E1-E2)/eta;
        if (a2 < L) 
            a2 = L;
        elseif (a2 > H) 
            a2 = H;
        end
    else
        %Lobj = objective function at a2=L
        %Hobj = objective function at a2=H
        f1 = (y1 * (E1 + B)) - (alph1 * k11) - (s * alph2 * k12);
        f2 = (y2 * (E2 + B)) - (s * alph1 * k12) - (alph2 * k22);
        L1 = alph1 + s * (alph2 - L);
        H1 = alph1 + s * (alph2 - H);
        Lobj = L1 * f1 + L * f2 + ((L1^2 * k11)/2) + ((L^2 * k22)/2) + s * L * L1 * k12;
        Hobj = H1 * f1 + H * f2 + ((H1^2 * k11)/2) + ((H^2 * k22)/2) + s * H * H1 * k12;

        % set new alpha2
        if (Lobj < Hobj-eps)
            a2 = L;
        elseif (Lobj > Hobj+eps)
            a2 = H;
        else
            a2 = alph2;
        end
    end
    
    if(a2<1e-8)
        a2 = 0;
    elseif(a2 > upperLim - 1e-8)
        a2 = upperLim;
    end

    if (abs(a2-alph2) < eps*(a2+alph2+eps))
        return;
    end
    a1 = alph1+s*(alph2-a2);

    %Update threshold to reflect change in Lagrange multipliers
    b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + B;
    b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + B;
    bold = B;
    B = (b1+b2)/2;
    
    %Update error cache using new Lagrange multipliers                
    % Update alpha values
    storedErrors = storedErrors + y1*(a1-alph1).*kern(idx1,:)+y2*(a2-alph2).*kern(idx2,:)+bold-B;
    storedErrors(idx1) = 0;
    storedErrors(idx2) = 0;
    alpha(idx1) = a1;
    alpha(idx2) = a2;
    result = true;
end

% Examine the training data
function result = examineExample(idx)
    global rowLength C alpha TARGET storedErrors upperLim;
    result = 0;
    y2 = TARGET(idx);
    alph2 = alpha(idx);
    if((alph2<C) || (alph2 > upperLim-C))
        E2 = SVMOutput(idx)-y2;
    else
        E2 = storedErrors(idx);
    end
    r2 = E2*y2;
    if ((r2 < -C) && (alph2 < upperLim) || (r2 > C) && (alph2 > 0))
        foundBounded = find((storedErrors > C) & (storedErrors < upperLim - C));
        if(~isempty(foundBounded) && (E2 > 0))
            [PP, idx1] = max(storedErrors);
            res = takeStep(idx1,idx);
            if(res)
                result = 1;
                return;
            end
        elseif(~isempty(foundBounded)) && (E2 < 0)
            [PP, idx1] = min(storedErrors);
            res = takeStep(idx1,idx);
            if(res)
                result = 1;
                return;
            end
        end
        if(~isempty(foundBounded))
            startPoint=randi(length(foundBounded));
            foundBounded=[foundBounded(startPoint:end) foundBounded(1:startPoint-1)];
            for idx1=foundBounded
                res = takeStep(idx1,idx);
                if(res)
                    result = 1;
                    return;
                end
            end
        end
        for idx1 = 1:rowLength
            res = takeStep(idx1,idx);
            if(res)
                result = 1;
                return;
            end
        end
    end
end

