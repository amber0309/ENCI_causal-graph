from __future__ import division
import numpy as np
from random import choice, randint
# from HSIC import hsic_gam, rbf_dot
from scipy.io import savemat, loadmat

class KEMDOPERATION:

	@staticmethod
	def kernel_embedding_K(dist, theta, delta):
		Len = len(dist)
		
		m,n = dist[0].shape 
		
		y = np.ones((m,n), dtype = float)
		
		for i in range(0, Len):
			
			d = dist[i]
			l = delta[0,i]
			# y1 = (1 + np.sqrt(3) * d/l) * np.exp(-np.sqrt(3) * d/l)
			# y2 = np.exp(-dist[i]/(delta[:,i])**2)
			# y3 = (1+np.sqrt(3) * d/l + 5*d*d/l/l)*np.exp(-np.sqrt(5)*d/l)
			# a = 0.5
			#y4 = (1+d*d/(2*a*l*l))**(-a)
			y2 = np.exp(-d**2/(l)**2)
		
			y = y * y2
			
		y = theta * y
		
		return y 
		
	@staticmethod
	def kernel_embedding_D(data, data_sr, feature_type):
	   
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
					elif Type == 'Categorical':
						dist_x1_x2 = float(x1[i]==x2[j])
					else:
						dist_x1_x2 = 0.0 
				
					K[k][i][j] = dist_x1_x2 
		return K 
		
	@staticmethod
	def median_dist(S1, S2, feature_type):
		L1 = len(S1[:,0])
		L2 = len(S2[:,0])
		num_of_feature = len(feature_type)
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			M = []
			for i in range(0, L2):
				for p in range(0, L1):
				
					if feature_type[t] == 'numeric':
						d = np.abs(S1[p,t] - S2[i,t])
					elif feature_type == 'Categorical':
						d = float(S1[p,t] == S2[i,t])
					else: 
						d = 0.0 
				
					M.append(d)
			MM[0,t] = np.median(M)
		return MM
		
	@staticmethod
	def mean_dist(S1, S2, feature_type):
		
		L = len(S1[:,0])
		num_of_feature = len(feature_type)
		MM = np.zeros((1, num_of_feature), dtype = float)
		for t in range(0, num_of_feature):
			M = []
			for i in range(0, L):
				for p in range(0, i):
				
					if feature_type[t] == 'numeric':
						d = np.abs(S1[p,t] - S2[i,t])
					elif feature_type == 'Categorical':
						d = float(S1[p,t] == S2[i,t])
					else: 
						d = 0.0 
				
					M.append(d)
			MM[0,t] = np.mean(M)
		return MM

def genX(sample_size, dim):
	ncoeff = 1

	wt = np.random.rand(3) + 0.5
	wt = wt/np.sum(wt)

	L1 = int(wt[0] * sample_size)
	x1 = 0.3 * np.random.randn(L1, dim) - 1
	L2 = int(wt[1] * sample_size)
	x2 = 0.3 * np.random.randn(L2, dim) + 1
	L3 = sample_size - L1 - L2
	x3 = 0.3 * np.random.randn(L3, dim)

	x = np.concatenate((x1, x2, x3), axis = 0)
	x = np.random.permutation(x)

	return x


def genY(x, label):
	ncoeff = 1
	sample_size = x.shape[0]
	dim = x.shape[1]
	
	c = 0.4 * np.random.rand(1) + 0.8

	wt = np.random.rand(3) + 0.5
	wt = wt/np.sum(wt)

	L1 = int(wt[0] * sample_size)
	n1 = 0.3 * np.random.randn(L1, dim) - 1
	L2 = int(wt[1] * sample_size)
	n2 = 0.3 * np.random.randn(L2, dim) + 1
	L3 = sample_size - L1 - L2
	n3 = 0.3 * np.random.randn(L3, dim)

	n = np.concatenate((n1, n2, n3), axis = 0)
	n = np.random.permutation(n)

	if label == 0:
		# y = np.sin(c * x) + n
		y = 1 / (x**2 + 1) + n * ncoeff
	elif label == 1:
		# y = np.sin(-c * x) + n
		# y = np.exp(c * x) + n
		# y = np.cos(c * x * n) + n
		y = np.sign(c * x) * ((c * x)**2) + n * ncoeff
	elif label == 2:
		y = np.cos(c * x * n) + n * ncoeff
	elif label == 3:
		y = np.sin(c * x) + n * ncoeff
	elif label == 4:
		y = x**2 + n * ncoeff
	elif label == 5:
		y = 2*np.sin(x) + 2*np.cos(x) + n * ncoeff
	elif label == 6:
		y = 4 * np.sqrt(np.abs(x)) + n * ncoeff
	else:
		pass

	return y

def Ato_mtpB(A):
	N = A.shape[0]
	[ridx, cidx] = np.nonzero(A)

	B = np.zeros((N,N))
	n_nonzero = len(ridx)

	for i in range(0,n_nonzero):
		cu_r = ridx[i]
		cu_c = cidx[i]

		B[N-1 - cu_r, N-1 - cu_c] = 1

	B = B.T

	return B


def genX(sample_size):
	wt = np.random.rand(3) + 0.5
	wt = wt/np.sum(wt)

	L1 = int(wt[0] * sample_size)
	x1 = 0.3 * np.random.randn(L1, 1) - 1
	L2 = int(wt[1] * sample_size)
	x2 = 0.3 * np.random.randn(L2, 1) + 1
	L3 = sample_size - L1 - L2
	x3 = 0.3 * np.random.randn(L3, 1)

	x = np.concatenate((x1, x2, x3), axis = 0)

	return x

def genY(x, label, ntype):
	ncoeff = 1
	sample_size = x.shape[0]
	
	c = 0.4 * np.random.rand(1) + 0.8

	if ntype == 'uni':
		n = np.random.rand(sample_size, 1)
	elif ntype == 'gmm':
		n = genX(sample_size)

	if label == 0:
		# y = np.sin(c * x) + n
		y = 1 / (x**2 + 1) + n * ncoeff
	elif label == 1:
		# y = np.sin(-c * x) + n
		# y = np.exp(c * x) + n
		# y = np.cos(c * x * n) + n
		y = np.sign(c * x) * ((c * x)**2) + n * ncoeff
	elif label == 2:
		y = np.cos(c * x * n) + n * ncoeff
	elif label == 3:
		y = np.sin(c * x) + n * ncoeff
	elif label == 4:
		y = x**2 + n * ncoeff
	elif label == 5:
		y = 2*np.sin(x) + 2*np.cos(x) + n * ncoeff
	elif label == 6:
		y = 4 * np.sqrt(np.abs(x)) + n * ncoeff
	else:
		pass

	return y

def genY_mltp(x, label, ntype):
	ncoeff = 1
	sample_size = x.shape[0]
	
	c = 0.4 * np.random.rand(1) + 0.8
	if ntype == 'uni':
		n = np.random.randn(sample_size, 1)
	elif ntype == 'gmm':
		n = genX(sample_size)

	if label == 0:
		# y = np.sin(c * x) + n
		y = 1 / (x**2 + 1) * n * ncoeff
	elif label == 1:
		# y = np.sin(-c * x) + n
		# y = np.exp(c * x) + n
		# y = np.cos(c * x * n) + n
		y = np.sign(c * x) * ((c * x)**2) * n * ncoeff
	elif label == 2:
		y = np.cos(c * x * n) * n * ncoeff
	elif label == 3:
		y = np.sin(c * x) * n * ncoeff
	elif label == 4:
		y = x**2 * n * ncoeff
	elif label == 5:
		y = (2*np.sin(x) + 2*np.cos(x)) * n * ncoeff
	elif label == 6:
		y = 4 * np.sqrt(np.abs(x)) * n * ncoeff
	else:
		pass

	return y
	
def pre_B(n):
	B = np.zeros((n, n))

	for i in range(1, n):
		idx = randint(0,i-1)
		B[i,idx] = 1

	return B


def pre_dlt_mltvar(XY, Z, nspl):
	# compute the delta for all k_xz and k_yz

	dim = XY[0].shape[1]
	dimz = Z[0].shape[1]
	feature_type = ['numeric']

	xyall = np.concatenate( XY, axis=0 )

	if dimz != 1:
		xyall = np.random.permutation(xyall)
		zall = xyall

	else:
		zall = np.concatenate( Z, axis=0)
		xyall = np.random.permutation(xyall)
		zall = np.random.permutation(zall)

	dlt = []
	for j in range(0, dim):
		if dimz == 1:
			dlt.append(KEMDOPERATION.median_dist(xyall[0:nspl,j].reshape(-1,1), zall[0:nspl,0].reshape(-1,1), feature_type))
		else:
			dlt.append(KEMDOPERATION.median_dist(xyall[0:nspl,j].reshape(-1,1), zall[0:nspl,j].reshape(-1,1), feature_type))

	return dlt

def gen_lingam(ngrp, md, g_sz, ntype):
	net_data = []

	# ---------- ----------
	# random generate skeleton
	# net_skltmat = pre_B(10)
	# net_skltmat = Ato_mtpB(net_skltmat)

	# fix skeleton
	net_skltmat = np.matrix('0 0 0 0 0 0;\
		0 0 0 0 0 0;\
		0 0 0 0 0 0;\
		1 0 0 0 0 0;\
		0 1 1 0 0 0;\
		0 0 0 1 1 0')
	# ---------- ----------

	nvar = net_skltmat.shape[0]
	# random skeleton
	# net_sklt = [-1] * nvar
	# net_sklt[1] = 0

	# for i in range(2,nvar):
	#   net_sklt[i] = randint(0,i-1)

	for gidx in range(0, ngrp):
		sample_size = choice(np.arange(g_sz, g_sz+10))
		if ntype == 'uni':
			net = np.random.rand(sample_size, 1)
		elif ntype == 'gmm':
			net = genX(sample_size)

		for vidx in range(1, nvar):
			# par = int(net_sklt[vidx, 0])
			par = net_skltmat[vidx,:].reshape(1,-1)
			[prow, rcol] = par.nonzero()
			if len(rcol) == 0:
				if ntype == 'uni':
					y = np.random.rand(sample_size, 1)
				elif ntype == 'gmm':
					y = genX(sample_size)
			else:
				y = np.zeros((sample_size, 1))
				for i in rcol:
					x = net[:,i].reshape(sample_size, 1)
					if md == 'ad':
						tempy = genY(x, randint(0,6), ntype)
					elif md == 'ml':
						tempy = genY_mltp(x, randint(0,6), ntype)

					y = y + tempy
			net = np.hstack((net, y))

		net_data.append(net)

	return (net_data, net_skltmat)

def pre_tensor(XY, w_coff):
	N_grp = len(XY)
	dim = XY[0].shape[1]

	feature_type = ['numeric']
	# Llist = []
	X_re = {}

	for k in range(0, N_grp):
		for didx in range(0, dim):
			XY[k][:,didx] = XY[k][:,didx] - np.mean(XY[k][:,didx])
			XY[k][:,didx] = XY[k][:,didx] / np.std(XY[k][:,didx])

	# print 'start median'
	dlt_list = pre_dlt_mltvar(XY, XY, 500)
	# print 'end of median'
	tau_x = np.zeros((dim, N_grp))

	# print 'start tau_x'

	for keridx in range(0, N_grp):
		# save k_x, k_y in lists
		xy = XY[keridx]
		L = xy.shape[0]

		H = np.identity(L) - np.ones((L)) / L

		for vidx in range(0, dim):
			# print 'kidx = %d, vidx = %d' %(keridx, vidx)
			x = xy[:,vidx].reshape(-1, 1)

			# del_xz = KEMDOPERATION.median_dist(x,z, feature_type)
			# del_yz = KEMDOPERATION.median_dist(y,z, feature_type)

			d_x = KEMDOPERATION.kernel_embedding_D(x, x, feature_type)

			k_x_i = KEMDOPERATION.kernel_embedding_K(d_x, 1, dlt_list[vidx]*w_coff)

			tau_xi = np.trace(np.dot(k_x_i, H))/ L / L

			tau_x[vidx, keridx] = tau_xi

			# Llist.append(L)

	# print 'finish tau_raw'
	for didx in range(0,dim):
		tau_x[didx,:] = tau_x[didx,:] - np.mean(tau_x[didx,:])

	X_re['X'] = tau_x
	return X_re

def net_chk(ngrp, md, wcf):
	"""
	ngrp        number of group
	md          generative mode
	wcf         coefficient before kernel width
	"""
	obs, Bori = gen_toy(ngrp, md)
	# obs, Bori = gen_lingam(500)
	# obs, Bori = gen_lgspr(500)
	X = pre_tensor(obs, wcf)
	re = X['X']
	# plt.imshow(re)
	# plt.show()

def net_pre(ngrp, md, wcf, g_sz, ntype):
	"""
	ngrp        number of group
	md          generative mode
	wcf         coefficient before kernel width
	g_sz        group size [g_sz, g_sz + 10]
	ntype       Gaussian noise / GMM noise
	"""
	# obs, Bori = gen_toy(ngrp, md, g_sz, ntype)
	obs, Bori = gen_lingam(ngrp, md, g_sz, ntype)
	# obs, Bori = gen_lgspr(500)

	X_re = pre_tensor(obs, wcf)
	X_re['Bori'] = Bori

	savemat('X_re', X_re )

def net_pre_lgm(ngrp, md, g_sz, ntype):
	obs, Bori = gen_toy(ngrp, md, g_sz, ntype)
	X = (np.concatenate( obs, axis=0 )).T

	X_re = {}
	X_re['X'] = X
	X_re['Bori'] = Bori

	savemat('X_re', X_re)

if __name__ == '__main__':
	# ----- generate net data for LiNGAM -----
	# net_pre_lgm(1000, 'ml', 40, 'gs')

	# ----- generate tau_x for LiNGAM -----
	# net_pre(2000, 'ml', 1.8, 40, 'gmm')

	# ----- case with fixed conditional distribution -----
	# expe(50)

	# ----- cased with diff conditional distribution -----
	# expe_dcon(50)
	x = loadmat('data1.mat')
	x = x['data1']
	y = loadmat('data2.mat')
	y = y['data2']
	z = loadmat('data3.mat')
	z = z['data3']

	feature_type = ['numeric']
	# dlt = KEMDOPERATION.median_dist(x, x, feature_type)
	# d_x = KEMDOPERATION.kernel_embedding_D(x, x, feature_type)
	# k_x = KEMDOPERATION.kernel_embedding_K(d_x, 1, dlt)
	data = [x, y, z]
	# dlt = pre_dlt_mltvar(data, data, 100)
	X = pre_tensor(data, 1)
	print X