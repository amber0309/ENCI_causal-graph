function [X, sklt] = gen_toy(ngrp, md, g_sz, ntype)

net_data = cell(1, ngrp);
net_sklt = [-1, 1, 2, 2];
nvar = length(net_sklt);
net_skltmat = [0,0,0,0; 1,0,0,0; 0,1,0,0; 0,1,0,0];

% for eidx = 1:(nvar-1)
%     par = int64(net_sklt(eidx));
%     net_skltmat(eidx, par) = 1;
% end

for gidx = 1:ngrp
    sample_size = randi(10) + g_sz;
    
    if ntype == 'uni'
        net = rand(sample_size, 1);
    elseif ntype == 'gmm'
        net = genX(sample_size);
    end
    
    for vidx = 2:nvar
        par = net_sklt(vidx);
        x = net(:,int64(par));
        if md == 'ad'
            y = genY(x, randi(6), ntype);
        elseif md == 'ml'
            y = genY_mltp(x, randi(6), ntype);
        end
        net = [net, y];
        
    net_data{gidx} = net;
    end
end

X = net_data;
sklt = net_skltmat;

end

function x = genX(sample_size)

wt = rand(3,1) + 0.5;
wt = wt/sum(wt);

L1 = floor(wt(1, 1) * sample_size);
x1 = 0.3 * randn(L1, 1) - 1;
L2 = floor(wt(2, 1) * sample_size);
x2 = 0.3 * randn(L2, 1) - 1;
L3 = sample_size - L1 - L2;
x3 = 0.3 * randn(L3, 1);

x = [x1; x2; x3];

end

function y = genY(x, label, ntype)

ncoeff = 1;
sample_size = size(x, 1);

c = 0.4*rand() + 0.8;

if ntype == 'uni'
    n = rand(sample_size, 1);
elseif ntype == 'gmm'
    n = genX(sample_size);
end

if label == 0
    y = 1 ./ (x.^2 + 1) + n * ncoeff;
elseif label == 1
    y = sign(c*x) .* ((c*x).^2) + n * ncoeff;
elseif label == 2
    y = cos(c * x .* n) + n * ncoeff;
elseif label == 3
    y = sin( c * x) + n * ncoeff;
elseif label == 4
    y = x.^2 + n * ncoeff;
elseif label == 5
    y = 2* sin(x) + 2*cos(x) + n * ncoeff;
elseif label == 6
    y = 4 * (abs(x).^(0.5)) + n * ncoeff;
end

end

function y = genY_mltp(x, label, ntype)

ncoeff = 1;
sample_size = size(x, 1);

c = 0.4*rand() + 0.8;

if ntype == 'uni'
    n = rand(sample_size, 1);
elseif ntype == 'gmm'
    n = genX(sample_size);
end

if label == 0
    y = 1 ./ (x.^2 + 1) .* n * ncoeff;
elseif label == 1
    y = sign(c*x) .* ((c*x).^2) .* n * ncoeff;
elseif label == 2
    y = cos(c * x .* n) .* n * ncoeff;
elseif label == 3
    y = sin( c * x) .* n * ncoeff;
elseif label == 4
    y = x.^2 .* n * ncoeff;
elseif label == 5
    y = (2* sin(x) + 2*cos(x)) .* n * ncoeff;
elseif label == 6
    y = 4 * (abs(x).^(0.5)) .* n * ncoeff;
end

end