function demo_HMM_Viterbi01
% Viterbi decoding in HMM to estimate best state sequence from observations.
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

addpath('./m_fcts/');


%% Parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model.nbStates = 5;
nbData = 10;
nbSamples = 5;


%% Load handwriting data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
demos=[];
load('data/2Dletters/C.mat');
%nbSamples = length(demos);
Data=[];
for n=1:nbSamples
	s(n).Data = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
	s(n).nbData = size(s(n).Data,2);
	Data = [Data s(n).Data]; 
end


%% Learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model = init_GMM_kmeans(Data, model);
model = init_GMM_kbins(Data, model, nbSamples);

% %Random initialization
% model.Trans = rand(model.nbStates,model.nbStates);
% model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
% model.StatesPriors = rand(model.nbStates,1);
% model.StatesPriors = model.StatesPriors/sum(model.StatesPriors);

%Uniform initialization
model.Trans = ones(model.nbStates,model.nbStates);
model.Trans = model.Trans ./ repmat(sum(model.Trans,2),1,model.nbStates);
model.StatesPriors = ones(model.nbStates,1);
model.StatesPriors = model.StatesPriors / sum(model.StatesPriors);

%Parameters refinement with EM
model = EM_HMM(s(1:end-1), model);

%MPE estimate of best path
H = computeGammaHMM(s(end), model);
[~,sMPE] = max(H);

%MAP estimate of best path (Viterbi)
sMAP = Viterbi_HMM(s(end).Data, model);


%% Plots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure('PaperPosition',[0 0 12 4],'position',[10,10,1300,500]); 
clrmap = lines(model.nbStates);
%Plot spatial data
subplot(2,2,[1,3]); hold on; axis off;
for i=1:model.nbStates
	plotGMM(model.Mu(:,i), model.Sigma(:,:,i), clrmap(i,:));
end
plot(Data(1,:), Data(2,:), '.', 'markersize', 12, 'color', [0 0 0]);
for t=1:nbData
	%plot(s(end).Data(1,t), s(end).Data(2,t), '.', 'markersize', 16, 'color', clrmap'*H(:,t));
	plot(s(end).Data(1,t), s(end).Data(2,t), '.', 'markersize', 12, 'color', clrmap(sMAP(t),:));
	plot(s(end).Data(1,t), s(end).Data(2,t), 'o', 'markersize', 6, 'color', clrmap(sMPE(t),:));
end
axis equal;
%Plot activation weights
subplot(2,2,2); hold on; box on;
for i=1:model.nbStates
	plot(H(i,1:nbData),'-','linewidth',2,'color',clrmap(i,:));
end
axis([1 nbData 0 1.05]); 
xlabel('t');
ylabel('\gamma_i');
set(gca,'xtick',1:nbData,'ytick',[0,1]);

%Plot state sequence
subplot(2,2,4); hold on; box on;
plot(sMAP,'.','markersize',12,'color',[.7 .7 .7]);
plot(sMPE,'o','markersize',6,'color',[0 0 0]);
hl = legend('s^{MAP}','s^{MPE}'); 
set(hl,'Box','off','Location','SouthEast');

axis([1 nbData 0 model.nbStates+0.5]); 
xlabel('t');
ylabel('s_i');
set(gca,'xtick',1:nbData,'ytick',1:model.nbStates);
%print('-dpng','-r300','graphs/demo_HMM_Viterbi01.png');

%Plot HMM trellis representation
figure('PaperPosition',[0 0 15 4],'position',[50,650,1200,400],'color',[1 1 1]);
axes('Position',[0 0 1 1]); hold on; axis off;
gridTrans = zeros(model.nbStates, model.nbStates, nbData);
gridNode = zeros(model.nbStates, nbData);
gridInit = zeros(model.nbStates,1);
for t=1:nbData
	gridNode(sMAP(t),t)=1;
	if t>1
		gridTrans(sMAP(t-1),sMAP(t),t-1)=1;
	end
end
plotHMMtrellis(model.Trans, model.StatesPriors, gridTrans, gridNode, gridInit);

%print('-dpng','-r300','graphs/demo_HMM_trellis04.png');
%pause;
%close all;

