function h = plotGMM(Mu, Sigma, color, valAlpha)
% This function displays the parameters of a Gaussian Mixture Model (GMM).
% Inputs -----------------------------------------------------------------
%   o Mu:           D x K array representing the centers of K Gaussians.
%   o Sigma:        D x D x K array representing the covariance matrices of K Gaussians.
%   o color:        3 x 1 array representing the RGB color to use for the display.
%   o valAlpha:     transparency factor (optional).
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


nbStates = size(Mu,2);
nbDrawingSeg = 35;
darkcolor = color*0.5; %max(color-0.5,0);
t = linspace(-pi, pi, nbDrawingSeg);

h=[];
for i=1:nbStates
	[V,D] = eig(Sigma(:,:,i));
	R = real(V*D.^.5);
	X = R * [cos(t); sin(t)] + repmat(Mu(:,i), 1, nbDrawingSeg);
	if nargin>3 %Plot with alpha transparency
		h = [h patch(X(1,:), X(2,:), color, 'lineWidth', 1, 'EdgeColor', darkcolor, 'facealpha', valAlpha,'edgealpha', valAlpha)];
		h = [h plot(Mu(1,:), Mu(2,:), '.', 'markersize', 6, 'color', darkcolor)];
	else %Plot without transparency
		h = [h patch(X(1,:), X(2,:), color, 'lineWidth', 1, 'EdgeColor', darkcolor)];
		h = [h plot(Mu(1,:), Mu(2,:), '.', 'markersize', 6, 'color', darkcolor)];
	end
end
