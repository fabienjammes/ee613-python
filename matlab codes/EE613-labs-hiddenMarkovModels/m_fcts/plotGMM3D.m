function h=plotGMM3D(Mu, Sigma, col1, alpha, dispOpt)
% Plot of GMMs in 3D
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

if nargin<4
	alpha=1;
end
if nargin<5
	dispOpt=1;
end

nbData = size(Mu,2);
nbPoints = 20; %nb of points to form a circular path
nbRings = 10; %Number of circular paths following the principal direction

pts0 = [cos(linspace(0,2*pi,nbPoints)); sin(linspace(0,2*pi,nbPoints))];

h=[];
for n=1:nbData
  [V0,D0] = eigs(Sigma(:,:,n));
  U0 = V0*D0^.5;

  ringpts0 = [cos(linspace(0,pi,nbRings+1)); sin(linspace(0,pi,nbRings+1))];
  ringpts = zeros(3,nbRings); 
  ringpts([2,3],:) = ringpts0(:,1:nbRings);
  U = zeros(3); 
  U(:,[2,3]) = U0(:,[2,3]); 
  ringTmp = U*ringpts; 
 
  %Compute touching circular paths
  for j=1:nbRings
    U = zeros(3); 
    U(:,1) = U0(:,1); 
    U(:,2) = ringTmp(:,j);
    pts = zeros(3,nbPoints); 
    pts([1,2],:) = pts0;
    xring(:,:,j) = U*pts + repmat(Mu(:,n),1,nbPoints);
  end

  %Plot filled ellispoid
  xringfull = xring;
  xringfull(:,:,end+1) = xringfull(:,end:-1:1,1); %Close the ellipsoid
  for j=1:size(xringfull,3)-1
    for i=1:size(xringfull,2)-1
      xTmp = [xringfull(:,i,j) xringfull(:,i+1,j) xringfull(:,i+1,j+1) xringfull(:,i,j+1) xringfull(:,i,j)];
			if dispOpt==1
				%Version 1 (for GMM plots)
				h = [h patch(xTmp(1,:),xTmp(2,:),xTmp(3,:), min(col1+0.1,1),'edgecolor',col1,'linewidth',1,'facealpha',alpha,'edgealpha',alpha)]; %,'facealpha',0.5
			else
				%Version 2 (for GMR plots)
				patch(xTmp(1,:),xTmp(2,:),xTmp(3,:), min(col1+0.35,1),'linestyle','none','facealpha',alpha);
			end
    end
  end
end
