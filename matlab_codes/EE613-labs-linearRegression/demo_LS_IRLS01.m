function demo_LS_IRLS01
% Iteratively reweighted least squares (IRLS)
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
nbData = 20; %Number of observations
p = 1; %L1 norm
nbIter = 20; %Number of iterations


%% Generate data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X = repmat(linspace(0,1,nbData)',1,nbVarIn); %Input data
Y = X * 5 + randn(nbData,nbVarOut)*2E-1; %Output data (with noise) %rand(nbVarIn,nbVarOut)
Y(4,:) = Y(4,:) + 5; %Simulation of outlier
Y(17,:) = Y(17,:) - 5; %Simulation of outlier


%% Regression
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initialization
W = eye(nbData);
A0 = X'*W*X \ X'*W * Y;
%Compute residuals of LS
r0 = 0;
for t=1:nbData
	r0 = r0 + norm(Y(t,:)-X(t,:)*A0, p);
end
	
%Iteratively reweighted least squares
for n=1:nbIter
	A = X'*W*X \ X'*W * Y;
	for t=1:nbData
		W(t,t) = abs(det(Y(t,:)-X(t,:)*A)^(p-2)); 
	end
	%Log of the residuals in IRLS
	rl(n) = 0;
	for t=1:nbData
		rl(n) = rl(n) + norm(Y(t,:)-X(t,:)*A, p);
	end
end
%Compute final residuals in IRLS
r = 0;
for t=1:nbData
	r = r + norm(Y(t,:)-X(t,:)*A, p);
end

disp(['Residuals with ordinary least squares               : ' num2str(r0)]);
disp(['Residuals with iteratively reweighted least squares : ' num2str(r)]);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 8 4],'position',[10,10,1200,500]);
%Normalize W for display purpose
W = W - min(diag(W));
W = W / max(diag(W));
i=1; j=1;
%LS
subplot(1,2,1); hold on; title(['Ordinary least squares (e=' num2str(r0,'%.1f') ')']);
for t=1:nbData
	plot([X(t,i) X(t,i)], [Y(t,j) X(t,i)*A0(i,j)], '-','linewidth',2,'color',[0 0 0]);
	plot(X(t,i), Y(t,j), '.','markersize',14,'color',[0 0 0]);
end
plot([0 1], [0 A0(i,j)], 'r-','linewidth',2);
xlabel(['x_' num2str(i)]); ylabel(['y_' num2str(j)]);
axis([0 1 -1 6]);
%IRLS
subplot(1,2,2); hold on; title(['Iteratively reweighted least squares (e=' num2str(r,'%.1f') ')']);
for t=1:nbData
	plot([X(t,i) X(t,i)], [Y(t,j) X(t,i)*A(i,j)], '-','linewidth',2,'color',ones(1,3)*(0.9-W(t,t)*0.9));
	plot(X(t,i), Y(t,j), '.','markersize',14,'color',[0 0 0]);
end
plot([0 1], [0 A(i,j)], 'r-','linewidth',2);
xlabel(['x_' num2str(i)]); ylabel(['y_' num2str(j)]);
axis([0 1 -1 6]);

% %Additional plot
% figure; hold on;
% plot(rl,'k-');
% plot([1,nbIter], [r0,r0], 'r-');
% xlabel('n'); ylabel('e');

%print('-dpng','graphs/demo_LS_IRLS01.png');
%pause;
%close all;
