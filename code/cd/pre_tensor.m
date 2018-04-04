function tau_x = pre_tensor(XY, w_coff)

N_grp = length(XY);
dim = size(XY{1}, 2);
feature_type = ['numeric'];

stand_xy = cell(1, N_grp);
for k = 1:N_grp
    stand_xy{k} = zscore(XY{k});
end

dlt_vec = pre_dlt_mltvar(stand_xy, stand_xy, 500);
tau_x = zeros(dim, N_grp);

for kidx = 1:N_grp
    xy = stand_xy{kidx};
    L = size(xy, 1);
    H = eye(L)-1/L*ones(L,L);
    
    for vidx = 1:dim
        x = xy(:,vidx);
        
        d_x = kernel_embedding_D(x, x, feature_type);
        k_x_i = kernel_embedding_K(d_x, 1, dlt_vec(1, vidx)*w_coff);
        tau_xi = trace( k_x_i * H )/L/L;
        tau_x(vidx, kidx) = tau_xi;
    end
end

tau_x = tau_x - repmat( mean(tau_x, 2),1, N_grp);

end