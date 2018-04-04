function [B, prc, rcl] = ENCI_graph(varargin)
%{
Input
1 input - the cell of all groups of data, X
2 input - a) cell of data, X
          b) true causal structure, square matrix B_true

Output
B       - the estimated causal strucutre
prc     - precision of causal structure estimation if B_true given
          -1 if B_true not given
rcl     - recall of causal structure estimation if B_true given
          -1 if B_true not given

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