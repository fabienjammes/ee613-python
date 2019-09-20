function demo_LS01
% Multivariate ordinary least squares 
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
nbVarIn = 2; %Dimension of input vector
nbVarOut = 1; %Dimension of output vector
nbData = 40; %Number of datapoints


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%A0 = rand(nbVarIn,nbVarOut)-0.5; %Linear relation between input and output (to be estimated)
A0 = [3; 2]; %Linear relation between input and output (to be estimated)
X = rand(nbData,nbVarIn); %Input data
Y = X * A0 + randn(nbData,nbVarOut)*5E-1; %Output data (with noise)


%% Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if nbData > nbVarIn
% 	A = (X'*X)\X' * Y; 
% else
% 	A = X'/(X*X') * Y; 
% end
A = pinv(X) * Y;

%Compute fitting error
e = 0;
for t=1:nbData
	%e = e + norm(Y(t,:)-X(t,:)*A)^2;
	e = e + (Y(t,:)-X(t,:)*A)' * (Y(t,:)-X(t,:)*A);
end
e

%norm(Y-X*A)^2
(Y-X*A)' * (Y-X*A)


%% 3D Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 8 4],'position',[10,10,800,700]); hold on; box on;
for t=1:nbData
	plot3([X(t,1) X(t,1)], [X(t,2) X(t,2)], [Y(t,1) X(t,:)*A], '-','linewidth',2,'color',[.7 .7 .7]);
	plot3(X(t,1), X(t,2), Y(t,1), '.','markersize',14,'color',[0 0 0]);
end
patch([0 0 1 1 0], [0 1 1 0 0], [0 [0,1]*A [1,1]*A [1,0]*A 0], [1 .4 .4],'linewidth',2,'edgecolor',[.8 0 0],'facealpha',0.2);
view(3); xlabel('x_1'); ylabel('x_2'); zlabel('y_1');
axis([0 1 0 1 -1 6]);

%print('-dpng','graphs/demo_LS01.png');
%pause;
%close all;
