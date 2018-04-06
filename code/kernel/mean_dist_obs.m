function MM = mean_dist(S1, S2, feature_type)

L1 = size(S1,1); % number of rows in S1
L2 = size(S2,1); % number of rows in S2
num_of_feature = size(feature_type,1); %number of rows in feature_type
tempMM = zeros(1,num_of_feature); %store median of each column
for t = 1:num_of_feature
    M = zeros(1, L1*L2);
    idx = 1;
    for i = 1:L2
        for p = 1:L1
            if strcmp(feature_type(t,:), 'numeric')
                % if S1(p,t) >= 1
                %     S1(p,t) = 1;
                % elseif S1(p,t) <= 0
                %     S1(p,t) = 0;
                % end
                
                % if S2(i,t) >= 1
                %     S2(i,t) = 1;
                % elseif S2(i,t) <= 0
                %     S2(i,t) = 0;
                % end
                
                d = abs(S1(p,t) - S2(i,t));
            elseif strcmp(feature_type(t,:), 'categorical')
                d = (S1(p,t) == S2(i,t));
            else
                d = 0;
            end
            M(1, idx) = d;
            idx = idx + 1;
        end
    end
    tempMM(1,t) = mean(M,2);
end
MM = tempMM;