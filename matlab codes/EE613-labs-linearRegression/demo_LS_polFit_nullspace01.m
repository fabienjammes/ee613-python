function demo_LS_polFit_nullspace01
% Polynomial fitting with least squares and nullspace projection
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
nbVarIn = 8; %Dimension of input vector
nbData = 4; %Number of datapoints
nbRepros = 8*5; %Number of reproductions


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%x = rand(nbData,1);
x = linspace(0,1,nbData)';
Y = rand(nbData,1);

X=[];
for i=0:nbVarIn-1
	X = [X, x.^i]; %-> X=[x.^3, x.^2, x, 1]
end

%Array used to display more points
xp = linspace(min(x),max(x),500)';
Xp=[];
for ii=0:nbVarIn-1
	Xp = [Xp, xp.^ii];
end


%% Regression with nullspace 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Nullspace projection operator - Version 1
N = eye(nbVarIn) - pinv(X)*X;

% %Nullspace projection operator - Version 2
% [U,S,V] = svd(pinv(X));
% sp = (sum(S,2)<1E-1); %Span of zero rows
% N = U(:,sp) * U(:,sp)'; %N = U * [0 0 0; 0 0 0; 0 0 1] * U'

for n=1:8
	for m=1:5
		v = zeros(nbVarIn,1); %[zeros(nbVarIn-1,1); n]
		v(n) = m*3;
		A = pinv(X) * Y + N * v;
		%Reconstruction
		r((n-1)*5+m).Yp = Xp*A;
	end
end

%Compute fitting error
e = norm(Y-X*A);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
i = min(2,nbVarIn); %Dimension to display for input data
j = 1; %Dimension to display for output data

figure('position',[10,10,1300,700]); hold on;
for n=1:nbRepros
	plot(Xp(:,i), r(n).Yp(:,j), '-','linewidth',1,'color',[.9 .9 .9]*rand(1,1)); %[.8 0 0]*n/nbRepros
end
plot(X(:,i), Y(:,j), '.','markersize',24,'color',[1 0 0]);
xlabel('x_1','fontsize',16);
ylabel('y_1','fontsize',16);
axis([min(X(:,i)) max(X(:,i)) min(r(end).Yp(:,j))-0.4 max(r(end).Yp(:,j))+0.4]);
set(gca,'xtick',[],'ytick',[]);

%print('-dpng','graphs/demo_LS_polFit_nullspace01.png');
%pause;
%close all;
