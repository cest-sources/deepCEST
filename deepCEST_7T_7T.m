% This code is most similar to the code used in DOI: 10.1002/mrm.27690 
% but here just using 7T spectra as input and 7T fit parameters as target
% in just a single slice of uploaded example data.
%
% Moritz Zaiss
% cest-sources.org 2021


clear
%% load example data from CEST eval git.
urlwrite('https://github.com/cest-sources/CEST_EVAL/blob/master/Example_fit.mat?raw=true','Example_fit.mat')
load('Example_fit.mat')

% load Lorentzfitfunction from CEST eval
urlwrite('https://github.com/cest-sources/CEST_EVAL/raw/master/levmar_fit/lorentzfit5pool.m','lorentzfit5pool.m')


%% generate input data X and target data T from Z-spectzra stack and from fitted parameter popt
tic
slices=1; % only 2D data here - please use much more data for actual training 
Z=Z_corrExt(:,:,slices,:);
X=reshape(Z,size(Z,1)*size(Z,2)*size(Z,3),size(Z,4))';
mask1= find(any(isnan(X)));

ZT=popt(:,:,slices,:);  
T=reshape(ZT,size(ZT,1)*size(ZT,2)*size(ZT,3),size(ZT,4))';
mask2= find(any(isnan(T)));

mask=union(mask1,mask2); % remove all NAN data
X(:,mask)=[];
T(:,mask)=[];

[X, T]=extend_data(X,T);  % this extends the input data by generating copies and adding noise, while keeping the same target data.
toc

% Create a Fitting Network
% hiddenLayerSize =10;
% net = fitnet(hiddenLayerSize);
hiddenLayerSize =[100 200 100];
net = feedforwardnet(hiddenLayerSize);

% Set up Division of Data for Training, Validation, Testing
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.performParam.normalization = 'standard';
net.performParam.regularization = 0.5;
net.trainFcn = 'trainscg';
net.trainParam.epochs=7000;
 
%% initialize weights with 0.1 % can eb done, doesnt have to be done. radnom initialization if this is left out
net.IW{1,1}=net.IW{1,1}*0+0.1;
net.LW{2,1}=net.LW{2,1}*0+0.1;
net.LW{3,2}=net.LW{3,2}*0+0.1;
net.LW{4,3}=net.LW{4,3}*0+0.1;
net.b{1}
net.b{2}=sort(net.b{2});
net.b{3}=sort(net.b{3});


%% Train the Network -  takes around 3000 iterations
tic 
% [net,tr] = train(net,X,T,'useParallel','yes','useGPU','yes','showResources','yes');  % faster in GPU
[net,tr] = train(net,X,T,'useParallel','yes','showResources','yes');
toc
 
% Test the Network
outputs = net(X);
% errors = gsubtract(outputs,T);
performance = perform(net,T,outputs)
 
% View the Network
% view(net)
genFunction(net,'DeepCEST_7T_7T_NN','MatrixOnly','yes');
% Plots
% Uncomment these lines to enable various plots.
 figure, plotperform(tr)
% figure, plottrainstate(tr)
% figure, plotfit(targets,outputs)
%  figure, plotregression(T,outputs)
% figure, ploterrhist(errors)

%% performance of Lorentzian amplitudes
figure, plotregression(T(5,:),outputs(5,:)); title('amides');
figure, plotregression(T(8,:),outputs(8,:)); title('NOE');  
figure, plotregression(T(11,:),outputs(11,:)); title('MT');
figure, plotregression(T(14,:),outputs(14,:)); title('amines');


%%  plot random Z-spectrum with fit and prediction of Lorentzian
idx=randi(size(X,2)); % random index
xx=X(:,idx);
tt=T(:,idx);
[Y] = DeepCEST_7T_7T_NN(xx);
L_NN = lorentzfit5pool(Y, P.SEQ.w, 1);
L_popt = lorentzfit5pool(tt, P.SEQ.w, 1);

figure,
subplot(2,3,1), plot(P.SEQ.w,xx,'m.', 'Displayname','input data'); ylim([0 1]); xlim([-10 10]); set(gca,'XDir','reverse');hold on;
plot(P.SEQ.w,L_NN,'r', 'Displayname','predicted Lorentzian');ylim([0 1]); xlim([-10 10]); set(gca,'XDir','reverse'); 
plot(P.SEQ.w,L_popt,'k', 'Displayname','original Lorentzian');ylim([0 1]); xlim([-10 10]); set(gca,'XDir','reverse');
legend('Location','southwest');
subplot(2,3,2), 
plot(P.SEQ.w,xx,'m.', 'Displayname','input data');hold on;
plot(P.SEQ.w,L_NN, 'Displayname','predicted Lorentzian');ylim([0 1]); xlim([-10 10]); set(gca,'XDir','reverse');
legend('Location','southwest');
subplot(2,3,3), 
plot(P.SEQ.w,xx,'m.', 'Displayname','input data'); hold on;
plot(P.SEQ.w,L_popt,'k', 'Displayname','original Lorentzian');ylim([0 1]); xlim([-10 10]); set(gca,'XDir','reverse');
legend('Location','southwest');

subplot(2,1,2), 
plot(P.SEQ.w,L_NN,'r', 'Displayname','predicted Lorentzian');ylim([0 1]); xlim([-10 10]); set(gca,'XDir','reverse'); hold on;
plot(P.SEQ.w,L_popt,'k', 'Displayname','original Lorentzian');ylim([0.3 1]); xlim([-10 10]); set(gca,'XDir','reverse');
legend('Location','southwest');


 