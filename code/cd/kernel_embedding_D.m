function K = kernel_embedding_D(data, data_sr, feature_type)

%data is a L * m_x matrix
%data_sr is a L * m_x matrix
%feature_type is a m_x * 1 vector

%output K is a 3d array (L,L,m_x)

len1 = size(data,1); %number of rows in data
len2 = size(data_sr,1); %number of rows in data_sr
num_of_feature = size(feature_type,1);

xx1 = data';
xx2 = data_sr';

tempK = zeros(len1,len2,num_of_feature);

for i = 1:len1
    for j = 1:len2
        for k = 1:num_of_feature
            Type = feature_type(k,:);
            x1 = xx1(k,:);
            x2 = xx2(k,:);
            
            if strcmp(Type, 'numeric')
                dist_x1_x2 = (x1(i) - x2(j));%^2;
            elseif strcmp(Type, 'categorical')
                dist_x1_x2 = (x1(i) == x2(j));
            else
                dist_x1_x2 = 0;
            end
            tempK(i,j,k) = dist_x1_x2;
        end
    end
end
K = tempK;