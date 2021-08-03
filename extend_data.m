function [XX,TT] = extend_data(X,T)

%% noise
% XX=[X, X+0.01*randn(size(X)), X+0.01*randn(size(X)), X+0.01*randn(size(X)), X+0.02*randn(size(X)), X+0.1*randn(size(X))];
% 
% TT=[T, T, T, T, T, T];

 XX=[X, X+0.01*randn(size(X)), X+0.01*randn(size(X))];
 
 TT=[T, T, T,];


%% sampling + interpolation
% 
% for ii=1:size(X,1)
%     
%     X(ii,:)
%     
% xq=linspace(-2,2,31);
% Zq=interp1(xx,Zx,xq);
%     
% end;


