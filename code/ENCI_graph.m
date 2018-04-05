function [B, prc, rcl] = ENCI_graph(varargin)
%{
Input (1 or 2 input arguments)
1 input  - the cell of all groups of data, each group is a matrix
           where rows corresponds to i.i.d sample and columns
           corresponds to random variables.
2 inputs - a) the cell of all groups of data, 
           b) true causal structure, square matrix B_true

Output
B       - the estimated causal strucutre
prc     - precision of causal structure estimation if B_true given
          -1 if B_true not given
rcl     - recall of causal structure estimation if B_true given
          -1 if B_true not given

Usage
1. [B, prc, rcl] = ENCI_graph(X, B_true)
2. [B, prc, rcl] = ENCI_graph(X)

Shoubo (shoubo.sub AT gmail.com)
05/04/2018
%}
addpath('cd');
narg = nargin;
X = varargin{1};

X_tensor = pre_tensor(X, 1);
[B, ~, ~, ~] = lingam( X_tensor );

if narg == 1
    prc = -1;
    rcl = -1;
else
    Bori = varargin{2};
    prc = length(find((Bori(:)~=0) & (B(:)~=0))) / length(find((B(:)~=0)));
    rcl = length(find((Bori(:)~=0) & (B(:)~=0))) / length(find((Bori(:)~=0)));
end

end