#! /usr/bin/env python
import numpy as np
import scipy.optimize as sc_opt
import sklearn.linear_model as sk_lm
#from tensor_utils import khatriRaoProd, tensor2mat
#from utils import multi_variate_normal

def khatriRaoProd(A, B):
	"""
	Computes the Khatri-Rao product of two matrices
	:param A: matrix (I, F)
	:param B: matrix (J, F)
	:return: Khatri-Rao product (IJ, F)
	"""
	I = A.shape[0]
	J = B.shape[0]
	F = A.shape[1]
	F1 = B.shape[1]

	if F != F1:
		print('A and B must have an equal number of columns.')

	KRprod = np.zeros((I*J, F))
	for f in range(0, F):
		KRprod[:, f] = np.dot(B[:, f][:, None], A[:, f][:, None].T).flatten()

	return KRprod

def tensor2mat(T, rows=None, cols=None):
	"""
	Matricization of a tensor
	:param T: tensor
	:param rows: dimensions that are rows of the final matrix
	:param cols: dimensions that are columns of the final matrix
	:return: matrix
	"""
	sizeT = np.array(T.shape)
	N = len(sizeT)

	if rows is None and cols is None:
		rows = range(0, N)

	if rows is None:
		if not isinstance(cols, list):
			cols = [cols]
		rows = list(set(range(0, N)) - set(cols))

	if cols is None:
		if not isinstance(rows, list):
			rows = [rows]
		cols = list(set(range(0, N)) - set(rows))

	T = np.transpose(T, rows + cols)

	return np.reshape(T, (np.prod(sizeT[rows]), np.prod(sizeT[cols])))


class MatrixRidgeRegression:
	"""
	Ridge regression for matrix-valued data
	Based on [Guo, Kotsia, Patras. 2012] and [Zhou, Li, Zhu. 2013]
	"""
	def __init__(self, rank):
		"""
		Initialize the model
		:param rank: rank of the weight matrices b1, b2
		"""
		self.b1 = []  # Left multiplying matrix
		self.b2 = []  # Right multiplying matrix
		self.alpha = []  # Constant term
		self.bVec = []
		self.rank = rank
		self.dY = 0

	def training(self, x, y, reg=1e-2, maxDiffCrit=1e-4, maxIter=200):
		"""
		Train the parameters of the MRR model
		:param x: input matrices (nb_data, dim1, dim2)
		:param y: output vectors (nb_data, dim_y)
		:param reg: regularization term
		:param maxDiffCrit: stopping criterion for the alternative least squares procedure
		:param maxIter: maximum number of iterations for the alternative least squares procedure
		:return:
		"""
		# Dimensions
		N = x.shape[0]
		d1 = x.shape[1]
		d2 = x.shape[2]
		self.dY = y.shape[1]

		for dim in range(0, self.dY):
			# Initialization
			# self.b1.append(np.random.randn(d1, self.rank))
			# self.b2.append(np.random.randn(d2, self.rank))
			# self.alpha.append(np.random.randn(1))
			self.b1.append(np.ones((d1, self.rank)))
			self.b2.append(np.ones((d2, self.rank)))
			self.alpha.append(np.zeros(1))
			self.bVec.append(np.random.randn(d1 * d2, 1))

			# Optimization of parameters (ALS procedure)
			nbIter = 1
			prevRes = 0

			while nbIter < maxIter:
				# Update b1
				zVec1 = np.zeros((N, d1*self.rank))
				for n in range(0, N):
					zVec1[n] = np.dot(x[n], self.b2[-1]).flatten()
				b1 = np.linalg.solve(zVec1.T.dot(zVec1) + np.eye(d1*self.rank)*reg, zVec1.T).dot(y[:, dim] - self.alpha[-1])
				self.b1[-1] = np.reshape(b1, (d1, self.rank))

				# Update b2
				zVec2 = np.zeros((N, d2*self.rank))
				for n in range(0, N):
					zVec2[n] = np.dot(x[n].T, self.b1[-1]).flatten()
				b2 = np.linalg.solve(zVec2.T.dot(zVec2) + np.eye(d2 * self.rank) * reg, zVec2.T).dot(y[:, dim] - self.alpha[-1])
				self.b2[-1] = np.reshape(b2, (d2, self.rank))

				# Update alpha
				self.bVec[-1] = np.dot(khatriRaoProd(self.b2[-1], self.b1[-1]), np.ones((self.rank, 1)))
				alpha = 0
				for n in range(0, N):
					alpha += y[n, dim] - np.dot(self.bVec[-1][:, None].T, x[n].flatten())
				self.alpha[-1] = alpha[0]/N

				# Compute residuals
				res = 0
				for n in range(0, N):
					res += (y[n, dim] - self.alpha[-1] - np.dot(self.bVec[-1][:, None].T, x[n].flatten()))**2

				resDiff = prevRes - res

				# Check convergence
				if resDiff < maxDiffCrit and nbIter > 1:
					print('MRR converged after %d iterations.' % nbIter)
					break
				nbIter += 1
				prevRes = res

			if resDiff > maxDiffCrit:
				print('MRR did not converged after %d iterations.' % nbIter)

	def testing(self, x):
		"""
		Estimate the output corresponding to a new input point
		:param x: new input data (dim1, dim2)
		:return: corresponding output prediction (dim_y)
		"""
		y = np.zeros(self.dY)
		for dim in range(0, self.dY):
			y[dim] = self.alpha[dim] + np.dot(self.bVec[dim][:, None].T, x.flatten())
		return y

	def testing_multiple(self, x):
		"""
		Estimate the outputs corresponding to new input points
		:param x: new input data (nb_data, dim1, dim2)
		:return: corresponding output predictions (nb_data, dim_y)
		"""
		N = x.shape[0]  
		y = np.zeros((N, self.dY))
		for n in range(0, N):
			for dim in range(0, self.dY):
				y[n, dim] = self.alpha[dim] + np.dot(self.bVec[dim][:, None].T, x[n].flatten())
		return y


class TensorRidgeRegression:
	"""
	Ridge regression for tensor-valued data
	Based on [Guo, Kotsia, Patras. 2012] and [Zhou, Li, Zhu. 2013]
	"""
	def __init__(self, rank):
		"""
		Initialize the model
		:param rank: rank of the weight matrices b1, b2
		"""
		self.W = []  # Regression coefficients
		self.alpha = []  # Constant term
		self.wVec = []
		self.rank = rank
		self.dY = 0

	def training(self, x, y, reg=1e-2, maxDiffCrit=1e-4, maxIter=200):
		"""
		Train the parameters of the MRR model
		:param x: input matrices (nb_data, dim1, dim2, ...)
		:param y: output vectors (nb_data, dim_y)
		:param reg: regularization term
		:param maxDiffCrit: stopping criterion for the alternative least squares procedure
		:param maxIter: maximum number of iterations for the alternative least squares procedure
		:return:
		"""
		# Dimensions
		N = x.shape[0]
		dX = x.shape[1:]
		self.dY = y.shape[1]

		for dim in range(0, self.dY):
			# Initialization
			wms = []
			for m in range(len(dX)):
				wms.append(np.ones((dX[m], self.rank)))

			self.alpha.append(np.zeros(1))
			self.wVec.append(np.reshape(np.zeros(dX), -1))

			# Optimization of parameters (ALS procedure)
			nbIter = 1
			prevRes = 0

			while nbIter < maxIter:
				for m in range(len(dX)):
					# Compute Wm complement (WM o ... o Wm+1 o Wm-1 o ... o W1)
					if m is 0:
						wmComplement = wms[1]
						for i in range(2, len(dX)):
							wmComplement = khatriRaoProd(wms[i], wmComplement)
					else:
						wmComplement = wms[0]
						for i in range(1, len(dX)):
							if i != m:
								wmComplement = khatriRaoProd(wms[i], wmComplement)

					# Update Wm
					zVec = np.zeros((N, dX[m] * self.rank))
					for n in range(0, N):
						zVec[n] = np.dot(tensor2mat(x[n], m), wmComplement).flatten()
					wm = np.linalg.solve(zVec.T.dot(zVec) + np.eye(dX[m]*self.rank)*reg, zVec.T).dot(y[:, dim] - self.alpha[-1])
					wms[m] = np.reshape(wm, (dX[m], self.rank))

				# Update alpha
				wTmp = khatriRaoProd(wms[1], wms[0])
				for i in range(2, len(dX)):
					wTmp = khatriRaoProd(wms[i], wTmp)

				self.wVec[-1] = np.dot(wTmp, np.ones((self.rank, 1)))
				alpha = 0
				for n in range(0, N):
					alpha += y[n, dim] - np.dot(self.wVec[-1][:, None].T, x[n].flatten())
				self.alpha[-1] = alpha[0]/N

				# Compute residuals
				res = 0
				for n in range(0, N):
					res += (y[n, dim] - self.alpha[-1] - np.dot(self.wVec[-1][:, None].T, x[n].flatten()))**2

				resDiff = prevRes - res

				# Check convergence
				if resDiff < maxDiffCrit and nbIter > 1:
					print('TRR converged after %d iterations.' % nbIter)
					break
				nbIter += 1
				prevRes = res

			if resDiff > maxDiffCrit:
				print('TRR did not converged after %d iterations.' % nbIter)

			self.W.append(wms)

	def testing(self, x):
		"""
		Estimate the output corresponding to a new input point
		:param x: new input data (dim1, dim2, ...)
		:return: corresponding output prediction (dim_y)
		"""
		y = np.zeros(self.dY)
		for dim in range(0, self.dY):
			y[dim] = self.alpha[dim] + np.dot(self.wVec[dim][:, None].T, x.flatten())
		return y

	def testing_multiple(self, x):
		"""
		Estimate the outputs corresponding to new input points
		:param x: new input data (nb_data, dim1, dim2, ...)
		:return: corresponding output predictions (nb_data, dim_y)
		"""
		N = x.shape[0]
		y = np.zeros((N, self.dY))
		for n in range(0, N):
			for dim in range(0, self.dY):
				y[n, dim] = self.alpha[dim] + np.dot(self.wVec[dim][:, None].T, x[n].flatten())
		return y
