from __future__ import division
import numpy as np
from kernels import KEMDOPERATION
from lingam import ICALiNGAM

class ENCI:
	def __init__(self, X, B=None):
		self.X = X 
		self.B_gt = B
		self.B_est = None

		self.n_domain = len(self.X)
		if isinstance(self.X[0], list):
			self.list_as_domain = True
			self.n_var = len(self.X[0])
		else:
			self.list_as_domain = False
			self.n_var = X[0].shape[1]

	def fit(self):
		# estimate the adjacency matrix using data in self.X
		# return precision and recall if self.B_gt is given

		while True:
			tau_X = self._trace_tensor()
			if np.sum( np.isnan(tau_X) ) == 0:
				break
			else:
				print('Matrix of traces contains NaN, reevaluating ...')
		md = ICALiNGAM()
		md.fit(tau_X)
		B_est = md.adjacency_matrix_
		self.B_est = B_est

		if self.B_gt is None:
			return B_est, -1 , -1
		else:
			prc, rcl = precision_recall(B_est, self.B_gt)
			return B_est, prc, rcl

	def _bandwidth_multivar(self, Z, nspl=500):
		# compute the kernel bandwidth for all variables

		while True:
			dlt = []

			if self.list_as_domain:
				for v in range(0, self.n_var):
					cu_data = self.X[0][v]
					for s in range(1, self.n_domain):
						cu_data = np.append( cu_data, self.X[s][v], axis=0 )
					cu_data = np.random.permutation(cu_data)
					if cu_data.shape[0] > nspl:
						n = nspl
					else:
						n = xall.shape[0]
					dlt.append( KEMDOPERATION.median_dist_scipy(cu_data[0:n], cu_data[0:n]) )
			else:
				dimz = Z[0].shape[1]
				xall = np.concatenate( self.X, axis=0 )

				if dimz != 1:
					xall = np.random.permutation(xall)
					zall = xall
				else:
					zall = np.concatenate( Z, axis=0)
					xall = np.random.permutation(xall)
					zall = np.random.permutation(zall)

				if xall.shape[0] > nspl:
					n = nspl
				else:
					n = xall.shape[0]

				for v in range(0, self.n_var):
					if dimz == 1:
						dlt.append(KEMDOPERATION.median_dist_scipy(xall[0:n, v].reshape(-1,1), zall[0:n, 0].reshape(-1,1) ))
					else:
						dlt.append(KEMDOPERATION.median_dist_scipy(xall[0:n, v].reshape(-1,1), zall[0:n, v].reshape(-1,1) ))

			if 0.0 in dlt:
				print('zero encountered in kernel bandwidth, reevaluating ...')
			else:
				break

		return dlt

	def _trace_tensor(self, w_coff=1):
		# compute the trace of tensors in RKHS for all variables

		dlt_list = self._bandwidth_multivar(self.X, 500)
		tau_x = np.zeros((self.n_var, self.n_domain), dtype=float)

		if self.list_as_domain:

			for s in range(0, self.n_domain):

				for v in range(0, self.n_var):
					cu_x = self.X[s][v]
					cu_L = cu_x.shape[0]
					H = np.identity(cu_L) - np.ones(cu_L)/cu_L

					d_x = KEMDOPERATION.kernel_embedding_D_scipy(cu_x, cu_x)
					k_x_i = KEMDOPERATION.kernel_embedding_K(d_x, 1, dlt_list[v]*w_coff)
					tau_xi = np.trace(np.dot(k_x_i, H))/ cu_L / cu_L
					tau_x[v, s] = tau_xi
		else:

			for s in range(0, self.n_domain):
				x = self.X[s]
				L = x.shape[0]

				H = np.identity(L) - np.ones((L)) / L

				for v in range(0, self.n_var):
					cu_x = x[:,v].reshape(-1, 1)

					d_x = KEMDOPERATION.kernel_embedding_D_scipy(cu_x, cu_x)
					k_x_i = KEMDOPERATION.kernel_embedding_K(d_x, 1, dlt_list[v]*w_coff)
					tau_xi = np.trace(np.dot(k_x_i, H))/ L / L
					tau_x[v, s] = tau_xi

		for v in range(0, self.n_var):
			tau_x[v,:] = tau_x[v,:] - np.mean(tau_x[v,:])

		return tau_x.T

def precision_recall(B, B_gt):
	prc = np.count_nonzero( B * B_gt ) / np.count_nonzero(B)
	rcl = np.count_nonzero( B * B_gt ) / np.count_nonzero(B_gt)
	return prc, rcl
