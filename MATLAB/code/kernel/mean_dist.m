function MM = mean_dist(S1, S2)

num_of_feature = size(S1,2); 
tempMM = zeros(1,num_of_feature); %store median of each column

for t = 1:num_of_feature
    dist = pdist2(S1(:,t), S2(:,t));
    dist(dist==0) = [];
    tempMM(1,t) = mean2(dist);
end

MM = tempMM;

end
