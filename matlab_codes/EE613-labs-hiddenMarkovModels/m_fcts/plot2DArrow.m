function h = plot2DArrow(pos,dir,col,lnw)
% Simple display of a 2D arrow
%
% Writing code takes time. Polishing it and making it available to others takes longer! 
% If some parts of the code were useful for your research of for a better understanding 
% of the algorithms, please reward the authors by citing the related publications, 
% and consider making your own research available in this way.
%
% @article{Calinon16JIST,
%   author="Calinon, S.",
%   title="A Tutorial on Task-Parameterized Movement Learning and Retrieval",
%   journal="Intelligent Service Robotics",
%   publisher="Springer Berlin Heidelberg",
%   doi="10.1007/s11370-015-0187-9",
%   year="2016",
%   volume="9",
%   number="1",
%   pages="1--29"
% }
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
	lnw = 2;
end

h(1) = plot([pos(1) pos(1)+dir(1)], [pos(2) pos(2)+dir(2)], '-','linewidth',lnw,'color',col);
sz = 1E-1;
pos = pos+dir;
if norm(dir)>sz
  dir = dir/norm(dir);
  prp = [dir(2); -dir(1)];
  dir = dir*sz;
  prp = prp*sz;
  msh = [pos pos-dir-prp/2 pos-dir+prp/2 pos];
  h(2) = patch(msh(1,:),msh(2,:),col,'edgecolor',col,'linewidth',max(lnw-3,.1));
end
