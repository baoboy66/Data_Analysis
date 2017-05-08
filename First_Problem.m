%x_axis = [18 12 7 5 22 15 6 21 19 18 18 10 16 14];
%y_axis = [34 36 12 11 14 29 14 15 31 40 37 20 21 25];
x_axis = [0];
y_axis = [0];
plot(x_axis, y_axis, 'k.');
axis([0,40,0,40]);
ylabel('Y values');
xlabel('X values');
%%
centroidX = [6 21.5 15.556];
centroidY = [12.333 14.5 29.222];
m = 3;
exponent = 2/(m-1);
total = 0;
for i = 1:3
    
    %eulidean distance
    %Need to manually input each centroid
   topx = x_axis - centroidX(1);
   topy = y_axis - centroidY(1);
   top = sqrt(topx.^2 + topy.^2);
   
   bottomx = x_axis - centroidX(i);
   bottomy = y_axis - centroidY(i);
   bottom = sqrt(bottomx.^2 + bottomy.^2);
   
   total = total + (top./bottom).^exponent;
end

membership = 1./total