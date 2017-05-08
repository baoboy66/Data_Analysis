data = [2 5 1 4 48 52 50 65 2 8 6 3 65 66 51 47 25 22 19 17 51 49 56 45 24 28 21 23 -2 -4 3 5;
       52 49 53 55 61 77 42 85 7 8 1 -2 12 3 9 10 61 68 72 47 26 28 19 31 8 9 2 -4 26 20 21 28];
   [c r] = size(data);
   target = zeros(1,r);
   target(1,17:32) = 1;
 plotpv(data, target);
   
   % input samples
X1=[rand(1,100);rand(1,100);ones(1,100)];   % class '+1'
X2=[rand(1,100);1+rand(1,100);ones(1,100)]; % class '-1'
X=[X1,X2];

% output class [-1,+1];
Y=[-ones(1,100),ones(1,100)];

% init weigth vector
w=[.5 .5 .5]';

% call perceptron
wtag=perceptron(X,Y,w);
% predict
ytag=wtag'*X;


% plot prediction over origianl data
figure;hold on
plot(X1(1,:),X1(2,:),'b.')
plot(X2(1,:),X2(2,:),'r.')

plot(X(1,ytag<0),X(2,ytag<0),'bo')
plot(X(1,ytag>0),X(2,ytag>0),'ro')
legend('class -1','class +1','pred -1','pred +1')