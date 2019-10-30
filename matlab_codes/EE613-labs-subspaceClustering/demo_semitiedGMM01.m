function demo_semitiedGMM01
% Semi-tied Gaussian Mixture Model by tying the covariance matrices of a
% Gaussian mixture model with a set of common basis vectors.
%
% Copyright (c) 2015 Idiap Research Institute, http://idiap.ch/
% Written by Ajay Tanwani and Sylvain Calinon
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

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 3; %Number of states in the GMM
model.nbVar = 3; %Number of variables [x1,x2,x3]
model.nbSamples = 5; %Number of demonstrations
model.params_Bsf = 5E-2; %Initial variance of B in semi-tied GMM
nbData = 300; %Length of each trajectory


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load('data/Zshape3D.mat'); %Load 'Data' 
model = init_GMM_kbins(Data, model, nbData);
model = EM_semitiedGMM(Data, model);


%% Plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('color',[1 1 1],'Position',[10 10 1400 650]);  
clrmap = lines(model.nbVar);

subplot(1,2,1); hold on; axis off;
plotGMM3D(model.Mu, model.Sigma+repmat(eye(model.nbVar)*2E0,[1,1,model.nbStates]), [0 .6 0]);
for n=1:model.nbSamples
	plot3(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), Data(3,(n-1)*nbData+1:n*nbData),'-','linewidth',1.5,'color',[.7 .7 .7]);
end
view(-40,6); axis equal;

subplot(1,2,2); hold on; axis off;
for n=1:model.nbSamples
	plot3(Data(1,(n-1)*nbData+1:n*nbData), Data(2,(n-1)*nbData+1:n*nbData), Data(3,(n-1)*nbData+1:n*nbData),'-','linewidth',1.5,'color',[.7 .7 .7]);
end
for i=1:model.nbVar
	mArrow3(zeros(model.nbVar,1), model.H(:,i), 'color',clrmap(i,:),'stemWidth',0.75, 'tipWidth',1.0); 
end
for i=1:model.nbStates
	for j=1:model.nbVar
		w = model.SigmaDiag(j,j,i).^0.5;
		if w>5E-1
			mArrow3(model.Mu(:,i), model.Mu(:,i)+model.H(:,j)*w, 'color',clrmap(j,:),'stemWidth',0.75, 'tipWidth',1.25);
		end
	end
end
view(-40,6); axis equal;

%print('-dpng','graphs/demo_semitiedGMM01.png');
%pause;
%close all;

