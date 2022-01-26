import torch 
import numpy as np
import scanpy as sc
import scvi
from torch.distributions import Normal
torch.set_default_dtype(torch.float64)

def nndsvd(A, k):
	'''
	This function implements the NNDSVD algorithm described in [1] for
	sinitializattion of Nonnegative Matrix Factorization Algorithms.

	Parameters
	------------

	 A    : the input nonnegative m x n matrix A
	 k    : the rank of the computed factors W,H
	 flag : indicates the variant of the NNDSVD Algorithm
			flag=0 --> NNDSVD
			flag=1 --> NNDSVDa
			flag=2 --> NNDSVDar

	Returns
	 -------------

	 W   : nonnegative m x k matrix
	 H   : nonnegative k x n matrix


	 References:

	 [1] C. Boutsidis and E. Gallopoulos, SVD-based initialization: A head
		 start for nonnegative matrix factorization, Pattern Recognition,
		 Elsevier

	'''

	#check the input matrix

	if A.min() < 0:
		raise ValueError('The input matrix contains negative elements !')


	#size of the input matirx
	m = A.size(0)
	n = A.size(1)

	#the matrices of the factorization
	try:
		W = torch.zeros((m,k)).cuda()
		H = torch.zeros((k,n)).cuda()
	except:
		W = torch.zeros((m,k))
		H = torch.zeros((k,n))



	#1st SVD --> partial SVD rank-k to the input matrix A.
	U, S, V = torch.svd(A)
	#print('U dtype',U.dtype)


	
	W[:,0] = torch.sqrt(S[0]) * torch.abs(U[:,0])
	H[0,:] = torch.sqrt(S[0]) * torch.abs(V[:,0])


	U = U[:,1:k]
	V = V[:,1:k]
	S = S[1:k]

	U_p = U.clone()
	U_p[U_p < 0] = 0

	U_n = U
	U_n[U_n > 0] = 0
	U_n = -U_n

	V_p = V.clone()
	V_p[V_p < 0] =0

	V_n = V
	V_n[V_n > 0] = 0
	V_n = -V_n

	norm_U_p = torch.norm(U_p,dim = 0)
	norm_U_n = torch.norm(U_n,dim = 0)
#     print('norm_U_p', norm_U_p)
#     print('norm_U_n',norm_U_n)


	norm_V_p = torch.norm(V_p,dim = 0)
	norm_V_n = torch.norm(V_n,dim = 0 )
#     print('norm_V_p',norm_V_p)
#     print('norm_V_n',norm_V_n)

	termp = norm_U_p * norm_V_p
	termn = norm_U_n * norm_V_n

	tmp_mul = torch.sqrt(S*termp)[None,:]
	W_p = tmp_mul * U_p / norm_U_p[None,:]
	H_p = tmp_mul * V_p / norm_V_p[None,:]
	H_p = H_p.T

	tmp_mul = torch.sqrt(S*termn)[None,:]
	W_n = tmp_mul * U_n / norm_U_n[None,:]
	H_n = tmp_mul * V_n / norm_V_n[None,:]
	H_n = H_n.T

	W[:,1:] = W_p
	H[1:,:] = H_p
	# print('W dtype',W.dtype)
	# print('W_n.dtype',W_n.dtype)
	ind_n = termp < termn
	W[:,1:][:,ind_n] = W_n[:,ind_n]
	H[1:,:][ind_n,:] = H_n[ind_n,:]
#     print(H[1:,:][ind_n,:] )
#     print(H_n[ind_n,:])

	eps = 0.0000000001

	W[W<eps] = 0
	H[H<eps] = 0

	return W, H



def dist2(x,c):
	'''
	Calculates the squared Euclidean distance between two matrices

	Parameters
	------------

	 x    : (M,N) matrix torch.tensor
	 c    : (L,N) matrix torch.tensor

	Returns
	 -------------
	res   : (M,L) matrix torch.tensor


	'''
	ndata,dimx = x.size()
	ncentres, dimc = c.size()

	if dimx != dimc:
		raise ValueError('Data dimension does not match dimension of centres')
	try:
		tmp_1 = torch.ones(ncentres,1).cuda()
		tmp_2 = torch.ones(ndata,1).cuda()
	except:
		tmp_1 = torch.ones(ncentres,1)
		tmp_2 = torch.ones(ndata,1)

	part1 = tmp_1 @ torch.sum(torch.square(x).T,0)[None,:]
	part2 = tmp_2 @  torch.sum(torch.square(c).T,0)[None,:]
	part3 = 2 * x@c.T

	del tmp_1, tmp_2
	torch.cuda.empty_cache()
	
   
	res = part1.T + part2 - part3

	return res 


def affinityMatrix(Diff,K = 20,sigma = 0.5):
	r"""
	Calculates affinity matrix given distance matrix

	Uses a scaled exponential similarity kernel to determine the weight of each
    edge based on `dist`. Optional hyperparameters `K` and `mu` determine the
    extent of the scaling (see `Notes`).

	Parameters
    ----------
    Diff : (N, N) array_like
        Distance matrix
    K : (0, N) int, optional
        Number of neighbors to consider. Default: 20
    mu : (0, 1) float, optional
        Normalization factor to scale similarity kernel. Default: 0.5

    Returns
    -------
    W : (N, N) torch.tenosr
        Affinity matrix


	Notes
    -----
    The scaled exponential similarity kernel, based on the probability density
    function of the normal distribution, takes the form:

    .. math::

       \mathbf{W}(i, j) = \frac{1}{\sqrt{2\pi\sigma^2}}
                          \ exp^{-\frac{\rho^2(x_{i},x_{j})}{2\sigma^2}}

    where :math:`\rho(x_{i},x_{j})` is the Euclidean distance (or other
    distance metric, as appropriate) between patients :math:`x_{i}` and
    :math:`x_{j}`. The value for :math:`\\sigma` is calculated as:

    .. math::

       \sigma = \mu\ \frac{\overline{\rho}(x_{i},N_{i}) +
                           \overline{\rho}(x_{j},N_{j}) +
                           \rho(x_{i},x_{j})}
                          {3}

    where :math:`\overline{\rho}(x_{i},N_{i})` represents the average value
    of distances between :math:`x_{i}` and its neighbors :math:`N_{1..K}`,
    and :math:`\mu\in(0, 1)\subset\mathbb{R}`.

	"""

	eps = 2.2204e-16
	Diff = (Diff + Diff.T)/2
	Diff = Diff - torch.diag(torch.diag(Diff))

	T = Diff.sort(dim = 1)[0]
	m, n = Diff.size()
	try: 
		W = torch.zeros((m,n)).cuda()
	except:
		W = torch.zeros((m,n))
	TT = T[:,1: K+1].mean(dim = 1) + eps
	Sig =  (TT[:,None] + TT[None,:] + Diff ) / 3

	Sig[Sig<=eps] = eps
	
	W = Normal(0 , sigma * Sig).log_prob(Diff).exp()
	del Sig, Diff
	torch.cuda.empty_cache()

	W = (W + W.T)/2

	return W


def SNF(*Wall, K = 20, t = 20, alpha = torch.tensor(1)):
	'''
	This function implements the SNF algorithm described in [1] for data integration.

	Parameters
	----------
	*aff : (N, N) array_like, torch.tensor
		Input similarity arrays; all arrays should be square and of equal size.
	K : (0, N) int, optional
		Hyperparameter normalization factor for scaling. Default: 20
	t : int, optional
		Number of iterations to perform information swapping. Default: 20
	alpha : (0, 1) torch.tensor, optional
		Hyperparameter normalization factor for scaling. Default: 1.0

	Returns
	-------
	W: (N, N) torch.tensor
		Fused similarity network of input arrays
	
	References:

	 [1] B Wang, A Mezlini, F Demir, M Fiume, T Zu, M Brudno, B Haibe-Kains, 
	 	 A Goldenberg (2014) Similarity Network Fusion: a fast and effective method 
	     to aggregate multiple data types on a genome wide scale. Nature Methods. 2014  
	
	
	'''
	Wall = torch.stack(list(Wall), dim=0)
	C = Wall.size(0)
	m,n = Wall[0].size()

	try:
		newW = torch.empty((C,m,n)).cuda()
	except:
		newW = torch.empty((C,m,n))

	for i in range(C):
		Wall[i] = Wall[i] / Wall[i].sum(1)[:,None]
		Wall[i] = (Wall[i] + Wall[i].T)/2
		newW[i] = FindDominateSet(Wall[i],K)


	Wsum = Wall.sum(dim = 0)
	# del newW
	# torch.cuda.empty_cache()

	for iter in range(t):
		#for i in range(C):
		Wall0 = newW @ (Wsum - Wall) @ newW.transpose(1,2) / (C - 1)
		Wall = BOnormalized(Wall0,alpha)
		Wsum = Wall.sum(0)
	

	del Wall0, Wall, newW
	torch.cuda.empty_cache()

	W = Wsum / C
	W = W / W.sum(1)[:,None]
	try:
		tmp = torch.eye(n).cuda()
	except:
		tmp = torch.eye(n)
	W = (W + W.T + tmp) / 2

	return W




def BOnormalized(W, alpha = torch.tensor(1)):

	"""
    Adds `alpha` to the diagonal of `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF
    alpha : (0, 1) torch.tensor, optional
        Factor to add to diagonal of `W` to increase subject self-affinity.
        Default: 1.0

    Returns
    -------
    W : (N, N) torch.tensor
        Normalized similiarity array
    """

	try:
		tmp = torch.eye(W[0].size(0)).cuda()
		alpha = alpha.cuda()
	except:
		tmp = torch.eye(W[0].size(0))

	W = W + alpha * tmp
	del tmp
	torch.cuda.empty_cache()

	W = (W +W.transpose(1,2)) / 2

	return W




def FindDominateSet(W,K = 20):
	"""
    Retains `K` strongest edges for each sample in `W`

    Parameters
    ----------
    W : (N, N) array_like
        Input data
    K : (0, N) int, optional
        Number of neighbors to retain. Default: 20

    Returns
    -------
    Wk : (N, N) torch.tensor
        Thresholded version of `W`
    """

	m,n = W.size()
	_,indices = torch.sort(W,1,descending = True)
	try:
		tmp = torch.arange(n)[:,None].cuda()
		newW = torch.zeros((m,n)).cuda()
	except:
		tmp = torch.arange(n)[:,None]
		newW = torch.zeros((m,n))
	#newW = torch.zeros((m,n))
	keeped_col = indices[:,:K]
	newW[tmp,keeped_col] = W[tmp,keeped_col]
	newW = newW / newW.sum(1)[:,None]

	return newW


def Wtrim(W0,K):
	'''
	Computes the KNN graph given a complete graph

	Parameters
    ----------
    W0: (N, N) torch.tensor, the complete graph
    K : (0, N) int, 
        Number of neighbors to retain.

    Returns
    -------
    W : (N, N) torch.tensor
        Trimmed version of W0

	'''
	n, m = W0.size()
	_, indices = torch.sort(W0, dim = 1, descending = True)

	try:
		tmp = torch.arange(n)[:,None].cuda()
		#W1 = torch.zeros((n,n)).cuda()
	except:
		tmp = torch.arange(n)[:,None]
		#W1 = torch.zeros((n,n))
	
	W0[tmp,indices[:,K:]] = 0
	#W1[tmp[0,:],indices[:,:K]] = W0[tmp[0,:],:K]
	W = (W0+ W0.T)/2

	return W



def nnlsm_blockpivot(A, B, is_input_prod=False, init=None):
	""" Nonnegativity-constrained least squares with block principal pivoting method and column grouping
	Solves min ||AX-B||_2^2 s.t. X >= 0 element-wise.
	J. Kim and H. Park, Fast nonnegative matrix factorization: An active-set-like method and comparisons,
	SIAM Journal on Scientific Computing, 
	vol. 33, no. 6, pp. 3261-3281, 2011.


	Parameters
	----------
	A : torch.tensor, shape (m,n)
	B : torch.tensor, shape (m,k)
	Optional Parameters
	-------------------
	is_input_prod : True/False. -  If True, the A and B arguments are interpreted as
			AtA and AtB, respectively. Default is False.
	init: torch.tensor, shape (n,k). - If provided, init is used as an initial value for the algorithm.
			Default is None.
	Returns
	-------
	X, (success, Y, num_cholesky, num_eq, num_backup)
	X : numpy.array, shape (n,k) - solution
	success : True/False - True if the solution is found. False if the algorithm did not terminate
			due to numerical errors.
	Y : numpy.array, shape (n,k) - Y = A.T * A * X - A.T * B
	num_cholesky : int - the number of Cholesky factorizations needed
	num_eq : int - the number of linear systems of equations needed to be solved
	num_backup: int - the number of appearances of the back-up rule. See SISC paper for details.
	"""
	if is_input_prod:
		AtA = A
		AtB = B
	else:
		AtA = A.T @ A
		# if sps.issparse(B):
		#     AtB = B.T.dot(A)
		#     AtB = AtB.T
		# else:
		AtB = A.T @ B

	(n, k) = AtB.shape
	MAX_ITER = n * 5

	if init is not  None:
		PassSet = init > 0
		#print('PassSet device:',PassSet.device)

		X, num_cholesky, num_eq = normal_eq_comb(AtA, AtB, PassSet)
		Y = AtA @ X - AtB
	else:
		try:
			X = torch.zeros((n, k)).cuda()
			PassSet = torch.zeros((n,k), dtype=torch.bool).cuda()
		except:
			X = torch.zeros((n, k))
			PassSet = torch.zeros((n,k), dtype=torch.bool)
		Y = -AtB
		#PassSet = np.zeros([n, k], dtype=bool)
		num_cholesky = 0
		num_eq = 0


	p_bar = 3
	try:
		p_vec = torch.zeros(k).cuda()
		ninf_vec = torch.zeros(k).cuda()
	except:
		p_vec = torch.zeros(k)
		ninf_vec = torch.zeros(k)

	p_vec[:] = p_bar
	#ninf_vec = np.zeros(k)
	ninf_vec[:] = n + 1
	not_opt_set = torch.logical_and(Y < 0, ~PassSet)
	infea_set =torch.logical_and(X < 0, PassSet)

	not_good = torch.sum(not_opt_set, dim=0) + torch.sum(infea_set, dim=0)
	not_opt_colset = not_good > 0
	not_opt_cols = not_opt_colset.nonzero()[:,0]

	big_iter = 0
	num_backup = 0
	success = True
	while not_opt_cols.numel() > 0:
		big_iter += 1
		if MAX_ITER > 0 and big_iter > MAX_ITER:
			success = False
			break

		cols_set1 = torch.logical_and(not_opt_colset, not_good < ninf_vec)
		temp1 = torch.logical_and(not_opt_colset, not_good >= ninf_vec)
		temp2 = p_vec >= 1
		cols_set2 = torch.logical_and(temp1, temp2)
		cols_set3 = torch.logical_and(temp1, ~temp2)

		cols1 = cols_set1.nonzero()[:,0]
		cols2 = cols_set2.nonzero()[:,0]
		cols3 = cols_set3.nonzero()[:,0]

		if cols1.numel() > 0:
			p_vec[cols1] = p_bar
			not_good = not_good.type(torch.float)
			# print('ninf_vec dtype:',ninf_vec.dtype)
			# print('not good dtype:',not_good.dtype)
			ninf_vec[cols1] = not_good[cols1]
			true_set = torch.logical_and(not_opt_set, torch.tile(cols_set1, (n, 1)))
			false_set = torch.logical_and(infea_set, torch.tile(cols_set1, (n, 1)))
			PassSet[true_set] = True
			PassSet[false_set] = False
		if cols2.numel() > 0:
			p_vec[cols2] = p_vec[cols2] - 1
			temp_tile = torch.tile(cols_set2, (n, 1))
			true_set = torch.logical_and(not_opt_set, temp_tile)
			false_set = torch.logical_and(infea_set, temp_tile)
			PassSet[true_set] = True
			PassSet[false_set] = False
		if cols3.numel() > 0:
			for col in cols3:
				candi_set = torch.logical_or(
					not_opt_set[:, col], infea_set[:, col])
				to_change = torch.max(candi_set.nonzero()[:,0])
				PassSet[to_change, col] = ~PassSet[to_change, col]
				num_backup += 1

		(X[:, not_opt_cols], temp_cholesky, temp_eq) = normal_eq_comb(
			AtA, AtB[:, not_opt_cols], PassSet[:, not_opt_cols])
		num_cholesky += temp_cholesky
		num_eq += temp_eq
		X[abs(X) < 1e-12] = 0
		Y[:, not_opt_cols] = AtA @ X[:, not_opt_cols]- AtB[:, not_opt_cols]
		Y[abs(Y) < 1e-12] = 0

		not_opt_mask = torch.tile(not_opt_colset, (n, 1))
		#print('not_opt_mask device:',not_opt_mask.device)
		not_opt_set = torch.logical_and(
			torch.logical_and(not_opt_mask, Y < 0), ~PassSet)
		infea_set = torch.logical_and(
			torch.logical_and(not_opt_mask, X < 0), PassSet)
		not_good = torch.sum(not_opt_set, dim=0) + torch.sum(infea_set, dim=0)
		not_opt_colset = not_good > 0
		not_opt_cols = not_opt_colset.nonzero()[:,0]

	return X, (success, Y, num_cholesky, num_eq, num_backup)


def normal_eq_comb(AtA, AtB, PassSet=None):
	""" Solve many systems of linear equations using combinatorial grouping.
	M. H. Van Benthem and M. R. Keenan, J. Chemometrics 2004; 18: 441-450
	Parameters
	----------
	AtA : torch.tensor, shape (n,n)
	AtB : torch.tensor, shape (n,k)
	Returns
	-------
	(Z,num_cholesky,num_eq)
	Z : torch.tensor, shape (n,k) - solution
	num_cholesky : int - the number of unique cholesky decompositions done
	num_eq: int - the number of systems of linear equations solved
	"""
	num_cholesky = 0
	num_eq = 0
	if AtB.numel() == 0:
		try:
			Z = torch.zeros([]).cuda()
		except:
			Z = torch.zeros([])
	elif (PassSet is None) or torch.all(PassSet):
		Z = torch.linalg.solve(AtA, AtB)
		num_cholesky = 1
		num_eq = AtB.shape[1]
	else:
		try:
			Z = torch.zeros(AtB.shape).cuda()
		except:
			Z = torch.zeros(AtB.shape)
		if PassSet.shape[1] == 1:
			if torch.any(PassSet):
				cols = PassSet.nonzero()[:,0]
				Z[cols] = torch.linalg.solve(AtA[torch.meshgrid(cols, cols)], AtB[cols])
				num_cholesky = 1
				num_eq = 1
		else:
			#
			# Both _column_group_loop() and _column_group_recursive() work well.
			# Based on preliminary testing,
			# _column_group_loop() is slightly faster for tiny k(<10), but
			# _column_group_recursive() is faster for large k's.
			#
			grps = _column_group_recursive(PassSet)
			for gr in grps:
				cols = PassSet[:, gr[0]].nonzero()[:,0]
				# print('cols device:',cols.device)
				# print('gr device:',gr.device)
				if cols.numel() > 0:
					ix1 = torch.meshgrid(cols, gr)
					#print('ix1 device:',ix1[0].device)
					ix2 = torch.meshgrid(cols, cols)
					#print('ix2 device:',ix2[0].device)
					#
					# scipy.linalg.cho_solve can be used instead of numpy.linalg.solve.
					# For small n(<200), numpy.linalg.solve appears faster, whereas
					# for large n(>500), scipy.linalg.cho_solve appears faster.
					# Usage example of scipy.linalg.cho_solve:
					# Z[ix1] = sla.cho_solve(sla.cho_factor(AtA[ix2]),AtB[ix1])
					#
					Z[ix1] = torch.linalg.solve(AtA[ix2], AtB[ix1])
					num_cholesky += 1
					num_eq += len(gr)
					num_eq += len(gr)
	return Z, num_cholesky, num_eq



def _column_group_recursive(B):
	""" Given a binary matrix, find groups of the same columns
		with a recursive strategy
	Parameters
	----------
	B : numpy.array, True/False in each element
	Returns
	-------
	A list of arrays - each array contain indices of columns that are the same.
	"""
	try:
		initial = torch.arange(0, B.shape[1]).cuda()
	except:
		initial = torch.arange(0, B.shape[1])
	return [a for a in column_group_sub(B, 0, initial) if len(a) > 0]


def column_group_sub(B, i, cols):
	vec = B[i][cols]
	if len(cols) <= 1:
		return [cols]
	if i == (B.shape[0] - 1):
		col_trues = cols[vec.nonzero()[:,0]]
		#print('col_trues device:',col_trues.device)
		col_falses = cols[(~vec).nonzero()[:,0]]
		#print('col_falses device:',col_falses.device)
		return [col_trues, col_falses]
	else:
		col_trues = cols[vec.nonzero()[:,0]]
		col_falses = cols[(~vec).nonzero()[:,0]]
		after = column_group_sub(B, i + 1, col_trues)
		after.extend(column_group_sub(B, i + 1, col_falses))
	return after



def compute_obj(X1,X2,S,W1,W2,H1,H2,D1,A1,D2,A2,alpha,gamma,theta1,theta2):
	''' function to comupte the objective to be optimized '''
	L1 = D1 - A1
	L2 = D2 - A2
	obj = torch.square(torch.norm(X1 - W1 @ H1.T))  + torch.square(torch.norm(X2 - W2 @ H2.T))
	obj = obj + 0.5 * alpha *( torch.square(torch.norm(S - H1@H1.T )) + torch.square(torch.norm(S-H2@H2.T)) )
	obj += gamma * (torch.trace(theta1.square() * H1.T @ L1 @ H1 + theta2.square() * H2.T @ L2 @ H2 ))

	return obj


def compute_obj_H(X1,X2,W1,W2,H,D1,A1,D2,A2,gamma,theta1,theta2):
	L1 = D1 - A1
	L2 = D2 - A2
	obj = torch.square(torch.norm(X1 - W1 @ H.T))  + torch.square(torch.norm(X2 - W2 @ H.T))
	obj += gamma * (torch.trace(theta1.square() * H.T @ L1 @ H + theta2.square() * H.T @ L2 @ H ))
	return obj



def compute_obj_3(X1,X2,X3, S,W1,W2,W3, H1,H2,H3, D1,A1,D2,A2,D3, A3,alpha,gamma,theta1,theta2, theta3):
	''' function to comupte the objective for three modalities case to be optimized '''
	L1 = D1 - A1
	L2 = D2 - A2
	L3 = D3 - A3
	obj = torch.square(torch.norm(X1 - W1 @ H1.T))  + torch.square(torch.norm(X2 - W2 @ H2.T)) + torch.square(torch.norm(X3 - W3 @ H3.T)) 
	obj = obj + 0.5 * alpha *( torch.square(torch.norm(S - H1@H1.T )) + torch.square(torch.norm(S-H2@H2.T)) + torch.square(torch.norm(S-H3@H3.T)))
	obj += gamma * (torch.trace(theta1.square() * H1.T @ L1 @ H1 + theta2.square() * H2.T @ L2 @ H2 + theta3.square() * H3.T @ L3 @ H3 ))
	return obj





def cos_opt(X):
	W  = X@X.T
	DX = torch.sqrt(torch.diag(W)) @ (torch.sqrt(torch.diag(W)).T)
	W = W/DX
	return W



def compute_obj_5(X11,X12,X21,X23,X31,X34,X41,X45,X51,X56,S1,S2,S3,S4,S5,W1,W2,W3,W4,W5,W6,H11,H12,H21,H23,H31,H34,H41,H45,H51,H56,D11,A11,D12,A12,D21,A21,D23,A23,D31,A31,D34,A34,D41,A41,D45,A45,D51,A51,D56,A56, alpha,gamma,theta11,theta12,theta21,theta23,theta31,theta34,theta41,theta45,theta51,theta56):
	L11 = D11-A11 
	L12 = D12-A12
	L21 = D21-A21
	L23 = D23-A23
	L31 = D31-A31
	L34 = D34-A34
	L41 = D41-A41
	L45 = D45-A45
	L51 = D51-A51
	L56 = D56-A56
	obj = torch.square(torch.norm(X11 - W1 @ H11.T)) + torch.square(torch.norm(X12 - W2 @ H12.T))
	obj += torch.square(torch.norm(X21 - W1 @ H21.T)) + torch.square(torch.norm(X23 - W3 @ H23.T))
	obj += torch.square(torch.norm(X31 - W1 @ H31.T)) + torch.square(torch.norm(X34 - W4 @ H34.T))
	obj += torch.square(torch.norm(X41 - W1 @ H41.T)) + torch.square(torch.norm(X45 - W5 @ H45.T))
	obj += torch.square(torch.norm(X51 - W1 @ H51.T)) + torch.square(torch.norm(X56 - W6 @ H56.T))
	
	tmp1 = torch.square(torch.norm(S1 - H11 @ H11.T)) + torch.square(torch.norm(S1 - H12 @ H12.T))
	tmp1 += torch.square(torch.norm(S2 - H21 @ H21.T)) + torch.square(torch.norm(S2 - H23 @ H23.T))
	tmp1 += torch.square(torch.norm(S3 - H31 @ H31.T)) + torch.square(torch.norm(S3 - H34 @ H34.T))
	tmp1 += torch.square(torch.norm(S4 - H41 @ H41.T)) + torch.square(torch.norm(S4 - H45 @ H45.T))
	tmp1 += torch.square(torch.norm(S5 - H51 @ H51.T)) + torch.square(torch.norm(S5 - H56 @ H56.T))
	obj += 0.5 * alpha * tmp1

	tmp2 = torch.trace(theta11**2*H11.T@L11@H11+theta12**2*H12.T@L12@H12+theta21**2*H21.T@L21@H21+theta23**2*H23.T@L23@H23)
	tmp2 += torch.trace(theta31**2*H31.T@L31@H31+theta34**2@H34.T@L34@H34+theta41**2*H41.T@L41@H41+theta45**2*H45.T@L45@H45+ theta51**2*H51.T@L51@H51+theta56**2*H56.T@L56@H56)
	obj += gamma * tmp2
	return obj