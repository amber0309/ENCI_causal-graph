from __future__ import division
import numpy as np
from scipy.spatial.distance import cdist

class KEMDOPERATION:

	@staticmethod
	def kernel_embedding_K(dist, theta, delta):
		"""
		compute the kernel matrix

		Input:
		dist,          - *[numpy.ndarray]* list of distance matrix
		               - each matrix is a (n_sample, n_feature) numpy.ndarray
		theta          - *float* coefficient for scaling the kernel matrix
		delta          - *[float]* list of kernel bandwidths of each dimension

		Output:
		K              - a list of numpy array
		                 each array is the distance matrix of a dimension
		"""
		L = len(dist)
		m, n = dist[0].shape 
		
		K = np.ones((m,n), dtype = float)
		for i in range(0, L):
			d = dist[i]
			l = delta[i]
			cu_K = np.exp(-d**2/((l)**2)) # RBF kernel
			K = K * cu_K

		return theta * K
		
	@staticmethod
	def kernel_embedding_D(data, data_sr, feature_type):
		"""
		compute the distance matrix of each dimension

		Input:
		data, data_sr  - *numpy.ndarray* (n_sample, n_feature) numpy arrays
		feature_type   - *[string]* type of data in each dimension ('numeric' or 'categorical')

		Output:
		K              - *[numpy.ndarray]* a list of numpy array
		                 each array is the distance matrix of a dimension
		"""
		len1  = len(data)
		len2 = len(data_sr)
		
		xx1 = np.transpose(data)
		xx2 = np.transpose(data_sr)

		temp = []
		for x in xx1:
			temp.append(x.tolist())
		xx1 = temp
		
		temp = []
		for x in xx2: 
		   temp.append(x.tolist())
		xx2 = temp
		
		num_of_feature = len(feature_type)
		K = []
		#print num_of_feature        
		for i in range(0, num_of_feature):
			K_k = np.zeros((len1, len2), dtype = float)
			K.append(K_k)
		
		dist_x1_x2 = 0.0 
		
		for i in range(0, len1):
			for j in range(0,len2):
				for k in range(0, num_of_feature):
				
					Type = feature_type[k]
					x1 = xx1[k]
					x2 = xx2[k]
				
					if Type == 'numeric':
						dist_x1_x2 = abs(x1[i] - x2[j])# ** 2 
					elif Type == 'categorical':
						dist_x1_x2 = float(x1[i]==x2[j])
					else:
						dist_x1_x2 = 0.0 
				
					K[k][i][j] = dist_x1_x2 
		return K 

	@staticmethod
	def kernel_embedding_D_scipy(data, data_sr):
		"""
		compute the distance matrix of each dimension
		!! works only for numerical arrays

		Input:
		data, data_sr  - *numpy.ndarray* (n_sample, n_feature) numpy arrays

		Output:
		D              - *[numpy.ndarray]* a list of numpy array
		                 each array is the distance matrix of a dimension
		"""
		n_feature = data.shape[1]
		D = []
		for t in range(0, n_feature):
			x_i = data[:,t].reshape(-1,1)
			y_i = data_sr[:,t].reshape(-1,1)
			d_i = cdist(x_i, y_i, 'euclidean')
			D.append(d_i)

		return D

	@staticmethod
	def median_dist(S1, S2, feature_type):
		"""
		compute the median pairwise distance of points in S1 and S2

		Input:
		S1, S2         - *numpy.ndarray* (n_sample, n_feature) numpy arrays
		feature_type   - *[string]* type of data in each dimension ('numeric' or 'categorical')

		Output:
		MM             - a (1, n_feature) numpy array
		"""
		L1 = len(S1[:,0])
		L2 = len(S2[:,0])
		n_feature = len(feature_type)
		MM = np.zeros((1, n_feature), dtype = float)
		for t in range(0, n_feature):
			M = []
			for i in range(0, L2):
				for p in range(0, i):
				
					if feature_type[t] == 'numeric':
						d = np.abs(S1[p,t] - S2[i,t])
					elif feature_type == 'categorical':
						d = float(S1[p,t] == S2[i,t])
					else: 
						d = 0.0 
				
					M.append(d)
			MM[0,t] = np.median(M)
		return MM
	
	@staticmethod
	def median_dist_np(S1, S2):
		"""
		compute the median pairwise distance of points in S1 and S2
		!! works only for numerical arrays

		Input:
		S1, S2         - *numpy.ndarray* (n_sample, n_feature) numpy arrays

		Output:
		M              - *numpy.ndarray* a (1, n_feature) numpy array
		"""
		n_feature = S1.shape[1]
		M = np.zeros((1, n_feature), dtype = float)
		for t in range(0, n_feature):
			b = np.array(np.meshgrid(S1[:,t], S2[:,t])).T.reshape(-1,2)
			abs_diff = abs(b[:,0] - b[:,1])
			M[0,t] = np.median(abs_diff[abs_diff != 0])
		return M

	@staticmethod
	def median_dist_scipy(S1, S2):
		"""
		compute the median pairwise distance of points in S1 and S2
		!! works only for numerical arrays

		Input:
		S1, S2         - *numpy.ndarray* (n_sample, n_feature) numpy arrays

		Output:
		M              - *numpy.ndarray* a (1, n_feature) numpy array
		"""
		n_feature = S1.shape[1]
		M = np.zeros((1, n_feature), dtype = float)
		for t in range(0, n_feature):
			x_i = S1[:,t].reshape(-1,1)
			y_i = S2[:,t].reshape(-1,1)
			d_i = cdist(x_i, y_i, 'euclidean')
			M[0,t] = np.median(d_i[d_i != 0])
		return M

	@staticmethod
	def mean_dist(S1, S2, feature_type):
		"""
		compute the mean pairwise distance of points in S1 and S2

		Input:
		S1, S2         - *numpy.ndarray* (n_sample, n_feature) numpy arrays
		feature_type   - *[string]* type of data in each dimension ('numeric' or 'categorical')

		Output:
		MM             - *numpy.ndarray* a (1, n_feature) numpy array
		"""
		L = len(S1[:,0])
		n_feature = len(feature_type)
		MM = np.zeros((1, n_feature), dtype = float)
		for t in range(0, n_feature):
			M = []
			for i in range(0, L):
				for p in range(0, i):
				
					if feature_type[t] == 'numeric':
						d = np.abs(S1[p,t] - S2[i,t])
					elif feature_type == 'categorical':
						d = float(S1[p,t] == S2[i,t])
					else: 
						d = 0.0 
				
					M.append(d)
			MM[0,t] = np.mean(M)
		return MM

	@staticmethod
	def mean_dist_np(S1, S2):
		"""
		compute the mean pairwise distance of points in S1 and S2
		!! works only for numerical arrays

		Input:
		S1, S2         - *numpy.ndarray* (n_sample, n_feature) numpy arrays

		Output:
		M              - *numpy.ndarray* a (1, n_feature) numpy array
		"""
		n_feature = S1.shape[1]
		M = np.zeros((1, n_feature), dtype = float)
		for t in range(0, n_feature):
			b = np.array(np.meshgrid(S1[:,t], S2[:,t])).T.reshape(-1,2)
			abs_diff = abs(b[:,0] - b[:,1])
			M[0,t] = np.mean(abs_diff[abs_diff != 0])
		return M

	@staticmethod
	def mean_dist_scipy(S1, S2):
		"""
		compute the mean pairwise distance of points in S1 and S2
		!! works only for numerical arrays

		Input:
		S1, S2         - *numpy.ndarray* (n_sample, n_feature) numpy arrays

		Output:
		M              - *numpy.ndarray* a (1, n_feature) numpy array
		"""
		n_feature = S1.shape[1]
		M = np.zeros((1, n_feature), dtype = float)
		for t in range(0, n_feature):
			x_i = S1[:,t].reshape(-1,1)
			y_i = S2[:,t].reshape(-1,1)
			d_i = cdist(x_i, y_i, 'euclidean')
			M[0,t] = np.mean(d_i[d_i != 0])
		return M
