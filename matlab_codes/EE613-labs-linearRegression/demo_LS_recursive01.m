function demo_LS_recursive01
% Recursive computation of least squares estimate (implementation with block data)
% 
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Sylvain Calinon, http://calinon.ch/
% 
% This file is part of PbDlib, http://www.idiap.ch/software/pbdlib/
% 
% PbDlib is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License version 3 as
% published by the Free Software Foundation.
% 
% PbDlib is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with PbDlib. If not, see <http://www.gnu.org/licenses/>.


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nbVarIn = 1; %Dimension of input vector
nbVarOut = 1; %Dimension of output vector
nbData = 200; %Number of observations


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A0 = rand(nbVarIn,nbVarOut); %Linear relation between input and output (to be estimated)
X = rand(nbData,nbVarIn); %Input data
Y = X * A0 + randn(nbData,nbVarOut)*1E-2; %Output data (with noise)


%% Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initial batch least squares estimate
iB = inv(X'*X);
A = iB * X' * Y;

%Arrival of new datapoints
nbDataNew = nbData/2; %Number of additional observations
V = rand(nbDataNew,nbVarIn); %New input data
C = V * rand(nbVarIn,nbVarOut) + randn(nbDataNew,nbVarOut)*1E-2; %New output data (C=V*A2 with noise)
Xnew = [X; V]; %Concatenated input data
Ynew = [Y; C]; %Concatenated output data

%Incremental update
tic
K =  iB*V' / (eye(nbDataNew) + V*iB*V'); %Kalman gain
Anew = A + K * (C-V*A); %Update A
%iB = iB - iB*V'/(1+V*iB*V')*V*iB; %Update iB
t=toc; disp(['Incremental update: ' num2str(t) ' sec.']);

%Batch update
tic
A0new = (Xnew'*Xnew)\Xnew' * Ynew;
t=toc; disp(['Batch update:       ' num2str(t) ' sec.']);

%Error on estimate
disp(['Difference of parameter estimates between batch and incremental: ' num2str(norm(A0new-Anew))]);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('position',[10,10,1300,800]); hold on;
i=1; j=1;
plot(X(:,i),Y(:,j), 'r.','markersize',20);
plot(V(:,i),C(:,j), 'g.','markersize',20);
plot([0 1],[0 A(i,j)], 'k--');
plot([0 1],[0 Anew(i,j)], 'k-');
xlabel(['x_' num2str(i)]); ylabel(['y_' num2str(j)]);
legend('Only red data','With green added');

%print('-dpng','graphs/demo_LS_recursive01.png');
%pause;
%close all;
