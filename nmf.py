import numpy as np

max_iter = 1000
def EuclideanDistanceUpdateRule(V, k, iters = max_iter):
	m, n = V.shape
	W = np.random.random((m,k))
	H = np.random.random((k,n))

	for _ in xrange(iters):
		for a in xrange(k):
			for mu in xrange(n):
				H[a,mu] = H[a,mu] * np.dot(W.T, V)[a, mu]/np.dot(np.dot(W.T, W), H)[a, mu]
			for i in xrange(m):
				W[i,a] = W[i,a] * np.dot(V, H.T)[i,a]/np.dot(np.dot(W, H), H.T)[i, a]
	return W, H

def DivergenceUpdateRule(V, k, iters = max_iter):
	m, n = V.shape
	W = np.random.random((m,k))
	H = np.random.random((k,n))
	for _ in xrange(iters):
		for a in xrange(k):
			for mu in xrange(n):
				H[a,mu] = H[a,mu] * sum( W[:,a] * V[:,mu]/np.dot(W,H)[:,mu])/sum(W[:,a])	
			for i in xrange(m):
				W[i,a] = W[i,a] * sum( H[a,:] * V[i,:] /np.dot(W,H)[i,:])/sum(H[a,:])
#			H[a,:]/sum(H[a,:])
#			W[:,a]/sum(W[:,a])
	return W, H

def fit(V, k, type = 0):
	if type == 0:
		return EuclideanDistanceUpdateRule(V, k)
	else:
		return DivergenceUpdateRule(V, k)

def rmse(A, B):
	import math
	return math.sqrt(((A - B) ** 2).mean(axis=None))

def normalize(X):
	rowSum = np.sum(X, axis=1)
	rowSum.shape = (rowSum.size,1)
	tmp = np.tile(rowSum,(1,X.shape[1]))
	return X/tmp

if __name__ == "__main__":
	V = np.array([[1.,2.,3.,4.],[5.,6.,7.,8.],[9.,10.,11.,12.]])

	W, H = fit(V, k=2, type=0)

	print("W = ")
	print(W)

	if np.any(W < 0.):
		print("Something worng with W matrix")

	print("H = ")
	print(H)
	if np.any(H < 0.):
		print("Something worng with H matrix")

	print("W * H = ")
	print(np.dot(W,H))

	print("RMSE = ")
	print(rmse(V, np.dot(W,H)))

	V2 = V.copy()
	V2 = normalize(V2)
	print(V2)
	W, H = fit(V, k=2, type=1)
	print("W = ")
	print(W)

	if np.any(W < 0.):
		print("Something worng with W matrix")

	print("H = ")
	print(H)
	if np.any(H < 0.):
		print("Something worng with H matrix")

	print("W * H = ")
	print(np.dot(W,H))
