from tkinter import W
import torch  
import numpy as np
import scanpy as sc
import scvi
from anndata import AnnData
from .utils import *
import time 
from scipy.sparse import isspmatrix
import bct
import umap,random
import matplotlib.pyplot as plt
import os
import pandas as pd


torch.set_default_dtype(torch.float64)

class JSNMF:
	'''
	JSNMF model. 

	Parameters
	----------
	RNA        : AnnData object that contains the data from RNA, true cell type should be
			      stored in RNA.obs['celltype']
	ATAC       : AnnData object that contains the data from ATAC, or other modality

	max_epochs : Number of max epochs to run, optinal, default is 200 epochs
 
	'''

	def __init__(self,
				 RNA: AnnData,
				 ATAC: AnnData,
				 max_epochs: int = 200):

		self.RNA = RNA
		self.ATAC = ATAC
		self.label = RNA.obs['celltype'].values

		self.max_epochs = max_epochs
		self.num_c  = num_c = len(np.unique(self.label))
		self.embedding = None
		try: 
			if  isspmatrix(RNA.X):
				self.X1 = torch.from_numpy(RNA.X.toarray()).T.cuda()
				self.X2 = torch.from_numpy(ATAC.X.toarray()).T.cuda()
			else:
				self.X1 = torch.from_numpy(RNA.X).T.cuda()
				self.X2 = torch.from_numpy(ATAC.X).T.cuda()

		except:
			if isspmatrix(RNA.X):
				self.X1 = torch.from_numpy(RNA.X.toarray()).T
				self.X2 = torch.from_numpy(ATAC.X.toarray()).T
			else:
				self.X1 = torch.from_numpy(RNA.X).T
				self.X2 = torch.from_numpy(ATAC.X).T


	def parameter_selection(self):
		'''
		Calculates the initilization values for the model. 

		Returns
		----------
		alpha   : init hyperparameter alpha
		beta    : init hyperparameter beta
		Inits   : dict stroing the init values

		'''
		Inits = {}
		num_c = self.num_c
		W1, H1 =  nndsvd(self.X1,num_c)
		print('nnsvd W1 done')
		W2, H2 = nndsvd(self.X2,num_c)
		print('nnsvd W2 done')

		Inits['W1'] = W1
		Inits['W2'] = W2
		Inits['H1'] = H1
		Inits['H2'] = H2

		D1 = dist2(self.X1.T,self.X1.T)
		print('D1 done')
		S1 = affinityMatrix(D1,20) # cell by cell
		print('S1 done')
		D2 = dist2(self.X2.T,self.X2.T)
		print('D2 done')
		S2 = affinityMatrix(D2,20) #cell by cell
		print('S2 done')


		Inits['A1'] = S1
		Inits['D1'] = torch.diag(S1.sum(0))
		Inits['L1'] = Inits['D1'] - Inits['A1']

		Inits['A2'] = S2
		Inits['D2'] = torch.diag(S2.sum(0))
		Inits['L2'] = Inits['D2'] - Inits['A2']

		print('SNF starts')
		W = SNF(S1,S2,K = 20)
		print('SNF done')
		Inits['S'] = W


		H1tH1 = H1.T @ H1
		H2tH2 = H2.T @ H2

		err1 = torch.square(torch.norm(self.X1 - W1@H1))
		err2 = torch.square(torch.norm(self.X2 - W2@H2))
		err3 = 0.5 * ( torch.square(torch.norm(Inits['S'] - H1tH1)) +  torch.square(torch.norm(Inits['S'] - H2tH2)))
		err4 = 0.5 * (H1@Inits['L1']@H1.T + H2@Inits['L2']@H2.T).trace()
		alpha = (err1 + err2) / err3
		gamma = torch.abs((err1 + err2) / err4)
		alpha = alpha / 10
		gamma = gamma / 10

		return alpha,gamma,Inits 


	def gcnmf(self):
		'''
		Main function to run the model. 

		Returns
		----------
        W1,W2,H1,H2    : resulting values of init  after running the model
		S			   : resulting complete graph

		use_epochs     : used epochs to converge

		objs		   : value of objective during each iteration

		'''
		Maxiter = self.max_epochs
		X1 = self.X1
		X2 = self.X2
		alpha = self.alpha
		gamma = self.gamma
		W1 = self.Inits['W1']
		W2 = self.Inits['W2']
		H1 = self.Inits['H1'].T
		H2 = self.Inits['H2'].T
		A1 = self.Inits['A1']
		D1 = self.Inits['D1']
		S = self.Inits['S']
		S = S / S.sum(dim =0,keepdim = True)
		A2 = self.Inits['A2']
		D2 = self.Inits['D2']
		n = S.size(0)

		try:
			theta1 = torch.tensor(1/2).cuda()
			theta2 = torch.tensor(1/2).cuda()
			yita1 = torch.tensor(1.).cuda()
			yita2 = torch.tensor(0.5).cuda()
			Lamda = torch.tenosor(0.5).cuda()
			obj_old = torch.tensor(1.).cuda()
			objs = torch.zeros((Maxiter,1)).cuda()
		except:
			theta1 = torch.tensor(1/2)
			theta2 = torch.tensor(1/2)
			yita1 = torch.tensor(1.)
			yita2 = torch.tensor(0.5)
			Lamda = torch.tensor(0.5)
			obj_old = torch.tensor(1.)
			objs = torch.zeros((Maxiter,1))


		for i in range(Maxiter):
			# update W1 W2 using bpp algorithm
			# W1, _ = nnlsm_blockpivot(H1, X1.T, False, W1.T) 
			# W2, _ = nnlsm_blockpivot(H2, X2.T, False, W2.T) 
			# W1 = W1.T
			# W2 = W2.T

			# updating w1 w2 via multiplication rule
			H1tH1 = H1.T @ H1
			H2tH2 = H2.T @ H2
			tmp1 = W1@H1tH1
			tmp1[tmp1 < 1e-10] = 1e-10
			W1 = W1 * (X1@H1) /tmp1
			tmp2 = W2@H2tH2
			tmp2[tmp2 < 1e-10] = 1e-10
			W2 = W2 * (X2@H2) / tmp2
			

			# update H1, H2
			W1tW1 = W1.T @ W1
			W2tW2 = W2.T @ W2
			H1tH1 = H1.T @ H1
			H2tH2 = H2.T @ H2
			tmp_deno_1 = gamma * theta1.square() * D1 @ H1 + (alpha + yita1) * H1@H1tH1 + H1@W1tW1
			tmp_deno_1[tmp_deno_1 < 1e-10] = 1e-10
			tmp_nume_1 = alpha * S.T@H1 + X1.T @ W1 + gamma * theta1.square() * A1@H1 + yita1 * H1
			H1 = H1 * (tmp_nume_1 / tmp_deno_1)

			tmp_deno_2 = gamma * theta2.square() * D2 @ H2 + (alpha + yita2) * H2@H2tH2 + H2@W2tW2
			tmp_deno_2[tmp_deno_2 < 1e-10] = 1e-10
			tmp_nume_2 = alpha * S.T@H2 + X2.T @ W2 + gamma * theta2.square() * A2@H2 + yita2 * H2
			H2 = H2 * (tmp_nume_2 / tmp_deno_2)

			# update S
			H1tH1 = H1@H1.T 
			H2tH2 = H2@H2.T
			Q = 1/2 * (H1tH1 + H2tH2)
			tmp = alpha*S + Lamda * S.sum(dim = 0, keepdim =True)
			tmp[tmp<1e-10] = 1e-10
			try: 
				tmp_ones = torch.ones((n,n)).cuda()
			except:
				tmp_ones = torch.ones((n,n))
			S = S * ((alpha * Q + Lamda * tmp_ones) / tmp)

			#update theta
			tmp1 = 1/torch.trace(H1.T@(D1 - A1)@H1)
			tmp2 = 1/torch.trace(H2.T@(D2 - A2)@H2)
			tmp = 1/torch.trace(H1.T@(D1 - A1)@H1) + 1/torch.trace(H2.T@(D2 - A2)@H2)
			theta1 = tmp1 / tmp
			theta2 = tmp2 / tmp

			

			#if stop_rule == 2:
			obj = compute_obj(X1,X2,S,W1,W2,H1,H2,D1,A1,D2,A2,alpha,gamma,theta1,theta2)
			objs[i,0] = obj
			error = torch.abs(obj_old - obj) / obj_old
			if  (error < 1e-5 and i > 0) or i == Maxiter - 1:
				print('number of epoch:', i+1)
				print('obj:',obj)
				print('converged!')
				break

			obj_old = obj

			print('number of epoch:', i+1)
			print('obj:',obj)

		S = (S + S.T)/2
		use_epochs = i+1

		return W1,W2,H1,H2,S,use_epochs,objs

	def run(self):
		'''
		Run the JSNMF model. Init time and main function time are recorded.

		Returns
		----------
		result  :  dict storing some information during the model running

		'''
		start = time.time()
		alpha, gamma, Inits = self.parameter_selection()
		end = time.time()
		self.init_t = end - start
		self.alpha = alpha
		self.gamma = gamma
		self.Inits = Inits

		print('Init done')

		start = time.time()
		W1,W2,H1,H2,S,used_epoch,objs  = self.gcnmf()
		end = time.time()
		self.run_t = end - start
		self.used_epoch = used_epoch

		result = dict(W1 = W1, W2 = W2,
					H1 = H1, H2 = H2,
					S = S, used_epoch = used_epoch,
					objs = objs,
					init_t = self.init_t,
					run_t  = self.run_t)

		self.result = result
		#return result

	def cluster(self, K = 50, step = 0.01, start = 2.7, upper = 4 ,seed = 3):

		'''
		Use louvain to cluster the cells based on the complete graph S, note that
		the function tries to find  a partition that has the same number of clusters
		as the true labels, the resolution parameter of louvain is found using binary search

		Parameters
		----------
		K      : (0, N) int, parameter of Wtrim
			     Number of neighbors to retain
		step   :  the step of binary search to find the partition, default 0.01
		start  :  start searching point of binary search
		upper  :  the upper bound of the reolution paramter to be searched
		seed   : seed parameter for louvain algorithm

		Returns
		-------
		res_clu : resulting cluster labels for each cell

		Note that sometimes exact number of clusters as true labels may not be found, paramters
		need to be adjusted then, like step, seed and upper

		'''
		S = self.result['S']
		A = Wtrim(S, K = 50)
		A = A.numpy()

		num_c = self.num_c

		tmp_gamma = start
		#use louvain to cluster based on A
		clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma, seed = seed)
		tmp_c = len(np.unique(clusters))
		tmp_clu = clusters

		res_clu = None


		# use binary search to find the corret gamma parameter
		while True:
			if tmp_c == num_c:
				res_clu = tmp_clu
				break

			if tmp_c < num_c:
				tmp_gamma = tmp_gamma + step
			else:
				tmp_gamma = tmp_gamma - step

			if tmp_gamma < 0 or tmp_gamma > upper:
				break
			tmp_gamma = round(tmp_gamma,2)
			#print(tmp_res)
			clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma,seed = seed)
			tmp_c = len(np.unique(clusters))
			tmp_clu = clusters

		return res_clu

	def visualize(self, label, tag = False, **kwarg):

		'''
		Visualize based on the complete graph S using Umap

		Parameters
		----------
		label     : array, true or clustered (louvain result) labels for each cell
		tag		  : if recalculte umap embedding
		**kwarg   : kwarg for the umap 	    

		Returns
		-------
		res_clu : resulting cluster labels for each cell

		'''


		# transfer S to distance matrix first
		S = self.result['S']
		data = 1-S
		data = data-np.diag(data.diagonal())
		reducer = umap.UMAP(**kwarg)

		# avoid recompute umap embedding
		if self.embedding is None:
			#min_dist = 0.68, n_neighbors=12
			embedding = reducer.fit_transform(data)
			self.embedding = embedding

		# recaculate embedding if needed
		if tag is True:
			embedding = reducer.fit_transform(data)
			self.embedding = embedding


		# plt.figure(figsize=(3, 1.5), dpi=300)
		# visualize
		for i in range(1,label.max() + 1):
			ind = label == i
			rgb = (random.random(), random.random() /2, random.random()/2)
			plt.scatter(self.embedding[ind, 0],self.embedding[ind, 1], s = 1.5, label = i,color = rgb)
		plt.legend(ncol=2,bbox_to_anchor=(1, 1.2))
		plt.show()

	def enrich_analysis(self, genes = None, peaks = None, topk_gene = 200, topk_peak = 1000,
							folder = 'h3k4'):
		'''
		Parameters
		----------
		genes     : gene names, if not none, save the factors for gene enrichment analysis
		peaks	  : peaks names, if not none, save the factors for region enrichment analysis
		topk_gene : number of top genes for each factor, default 200
		topk_peak : number of top peak for each factor, default 1000
		folder    : the name of the folder to save the results

		Returns
		-------
		saved files containing factors for  
		
		'''
		
		
		if genes is not None:
			path = folder +'/W1'
			if not os.path.exists(path):
				os.makedirs(path)
			W1 = self.result['W1']
			nfactor = W1.shape[1]
			sorted, indices = torch.sort(W1,dim = 0,descending=True)
			print('Start writing factors for gene enrichment analysis')
			for i in range(nfactor):
				tmp_dict = {'gene symbol':genes[indices[:topk_gene,i]]}
				tmp_pd = pd.DataFrame(tmp_dict)
				tmp_pd.to_csv(folder + '/W1/factor_' + str(i+1) + '.csv',index = False)
				print('Successfully write factor_' + str(i+1) + ' for gene enrichment analysis')
			print('done')
			print()


		if peaks is not None:
			path = folder +'/W2'
			if not os.path.exists(path):
				os.makedirs(path)
			W2 = self.result['W2']
			nfactor = W2.shape[1]
			sorted, indices = torch.sort(W2,dim = 0,descending=True)
			print('Start writing factors for region enrichment analysis')
			for i in range(nfactor):
				tmp_dict = {'peak loc':peaks[indices[:topk_peak,i]]}
				tmp_pd = pd.DataFrame(tmp_dict)
				tmp_pd.to_csv(folder + '/W2/factor_' + str(i+1) + '.csv')
				print('Successfully write factor_' + str(i+1) + ' for region enrichment analysis')
			
			print('done')




class JSNMF_3mod:


	def __init__(self,
				 X1: torch.tensor,
				 X2: torch.tensor,
				 X3: torch.tensor,
				 label,
				 max_epochs: int = 200):

		try: 
			self.X1 = X1.cuda()
			self.X2 = X2.cuda()
			self.X3 = X3.cuda()
		except:
			self.X1 = X1
			self.X2 = X2
			self.X3 = X3
		
		self.label = label

		self.max_epochs = max_epochs
		self.num_c  = num_c = len(np.unique(self.label))
		self.embedding = None



	def parameter_selection(self):
		'''
		Calculates the initilization values for the model. 

		Returns
		----------
		alpha   : init hyperparameter alpha
		beta    : init hyperparameter beta
		Inits   : dict stroing the init values

		'''
		Inits = {}
		num_c = self.num_c
		W1, H1 =  nndsvd(self.X1,num_c)
		print('nnsvd W1 done')
		W2, H2 = nndsvd(self.X2,num_c)
		print('nnsvd W2 done')
		W3, H3 = nndsvd(self.X3,num_c)
		print('nnsvd W3 done')

		Inits['W1'] = W1
		Inits['W2'] = W2
		Inits['W3'] = W3
		Inits['H1'] = H1
		Inits['H2'] = H2
		Inits['H3'] = H3


		D1 = dist2(self.X1.T,self.X1.T)
		print('D1 done')
		S1 = affinityMatrix(D1,20) # cell by cell
		print('S1 done')

		D2 = dist2(self.X2.T,self.X2.T)
		print('D2 done')
		S2 = affinityMatrix(D2,20) #cell by cell
		print('S2 done')

		D3 = dist2(self.X3.T,self.X3.T)
		print('D3 done')
		S3 = affinityMatrix(D3,20) #cell by cell
		print('S3 done')



		Inits['A1'] = S1
		Inits['D1'] = torch.diag(S1.sum(0))
		Inits['L1'] = Inits['D1'] - Inits['A1']

		Inits['A2'] = S2
		Inits['D2'] = torch.diag(S2.sum(0))
		Inits['L2'] = Inits['D2'] - Inits['A2']

		Inits['A3'] = S3
		Inits['D3'] = torch.diag(S3.sum(0))
		Inits['L3'] = Inits['D3'] - Inits['A3']






		print('SNF starts')
		W = SNF(S1,S2, S3, K = 20)
		print('SNF done')
		Inits['S'] = W


		H1tH1 = H1.T @ H1
		H2tH2 = H2.T @ H2
		H3tH3 = H3.T @ H3

		err1 = torch.square(torch.norm(self.X1 - W1@H1))
		err2 = torch.square(torch.norm(self.X2 - W2@H2))
		err21 = torch.square(torch.norm(self.X3 - W3@H3))
		err3 = 1/3 * ( torch.square(torch.norm(Inits['S'] - H1tH1)) +  torch.square(torch.norm(Inits['S'] - H2tH2))  +  torch.square(torch.norm(Inits['S'] - H3tH3)) )
		err4 = 1/3 * (H1@Inits['L1']@H1.T + H2@Inits['L2']@H2.T   + H3@Inits['L3']@H3.T     ).trace()
		alpha = (err1 + err2  + err21) / err3
		gamma = torch.abs((err1 + err2 + err21) / err4)
		alpha = alpha / 10
		gamma = gamma / 10







		return alpha,gamma,Inits 



	def gcnmf(self):
		'''
		Main function to run the model. 

		Returns
		----------
        W1,W2,H1,H2    : resulting values of init  after running the model
		S			   : resulting complete graph

		use_epochs     : used epochs to converge

		objs		   : value of objective during each iteration

		'''
		Maxiter = self.max_epochs
		X1 = self.X1
		X2 = self.X2
		X3 = self.X3
		alpha = self.alpha
		gamma = self.gamma
		W1 = self.Inits['W1']
		W2 = self.Inits['W2']
		W3 = self.Inits['W3']
		H1 = self.Inits['H1'].T
		H2 = self.Inits['H2'].T
		H3 = self.Inits['H3'].T
		A1 = self.Inits['A1']
		D1 = self.Inits['D1']
		S = self.Inits['S']
		S = S / S.sum(dim =0,keepdim = True)
		A2 = self.Inits['A2']
		D2 = self.Inits['D2']
		A3 = self.Inits['A3']
		D3 = self.Inits['D3']

		n = S.size(0)

		try:
			theta1 = torch.tensor(1/2).cuda()
			theta2 = torch.tensor(1/2).cuda()
			theta3 = torch.tensor(1/3).cuda()
			yita1 = torch.tensor(1.).cuda()
			yita2 = torch.tensor(0.5).cuda()
			yita3 = torch.tensor(0.5).cuda()
			Lamda = torch.tenosor(0.5).cuda()
			obj_old = torch.tensor(1.).cuda()
			objs = torch.zeros((Maxiter,1)).cuda()
		except:
			theta1 = torch.tensor(1/3)
			theta2 = torch.tensor(1/3)
			theta3 = torch.tensor(1/3)
			yita1 = torch.tensor(1.)
			yita2 = torch.tensor(0.5)
			yita3 = torch.tensor(0.5)
			Lamda = torch.tensor(0.5)
			obj_old = torch.tensor(1.)
			objs = torch.zeros((Maxiter,1))


		for i in range(Maxiter):
			# update W1 W2 using bpp algorithm
			# W1, _ = nnlsm_blockpivot(H1, X1.T, False, W1.T) 
			# W2, _ = nnlsm_blockpivot(H2, X2.T, False, W2.T) 
			# W3, _ = nnlsm_blockpivot(H3, X3.T, False, W3.T) 
			# W1 = W1.T
			# W2 = W2.T
			# W3 = W3.T

			# updating w1 w2 via multiplication rule
			H1tH1 = H1.T @ H1
			H2tH2 = H2.T @ H2
			tmp1 = W1@H1tH1
			tmp1[tmp1 < 1e-10] = 1e-10
			W1 = W1 * (X1@H1) /tmp1
			tmp2 = W2@H2tH2
			tmp2[tmp2 < 1e-10] = 1e-10
			W2 = W2 * (X2@H2) / tmp2
			tmp3 = W3@H3tH3
			tmp3[tmp3 < 1e-10] = 1e-10
			W3 = W3 * (X3@H3) / tmp3

			# update H1, H2
			W1tW1 = W1.T @ W1
			W2tW2 = W2.T @ W2
			W3tW3 = W3.T @ W3
			H1tH1 = H1.T @ H1
			H2tH2 = H2.T @ H2
			H3tH3 = H3.T @ H3

			tmp_deno_1 = gamma * theta1.square() * D1 @ H1 + (alpha + yita1) * H1@H1tH1 + H1@W1tW1
			tmp_deno_1[tmp_deno_1 < 1e-10] = 1e-10
			tmp_nume_1 = alpha * S.T@H1 + X1.T @ W1 + gamma * theta1.square() * A1@H1 + yita1 * H1
			H1 = H1 * (tmp_nume_1 / tmp_deno_1)

			tmp_deno_2 = gamma * theta2.square() * D2 @ H2 + (alpha + yita2) * H2@H2tH2 + H2@W2tW2
			tmp_deno_2[tmp_deno_2 < 1e-10] = 1e-10
			tmp_nume_2 = alpha * S.T@H2 + X2.T @ W2 + gamma * theta2.square() * A2@H2 + yita2 * H2
			H2 = H2 * (tmp_nume_2 / tmp_deno_2)

			tmp_deno_3 = gamma * theta3.square() * D3 @ H3 + (alpha + yita3) * H3@H3tH3 + H3@W3tW3
			tmp_deno_3[tmp_deno_3 < 1e-10] = 1e-10
			tmp_nume_3 = alpha * S.T@H3 + X3.T @ W3 + gamma * theta3.square() * A3@H3 + yita3 * H3
			H3 = H3 * (tmp_nume_3 / tmp_deno_3)





			# update S
			H1tH1 = H1@H1.T 
			H2tH2 = H2@H2.T
			H3tH3 = H3@H3.T
			Q = 1/3 * (H1tH1 + H2tH2 + H3tH3)
			tmp = alpha*S + Lamda * S.sum(dim = 0, keepdim =True)
			tmp[tmp<1e-10] = 1e-10
			try: 
				tmp_ones = torch.ones((n,n)).cuda()
			except:
				tmp_ones = torch.ones((n,n))
			S = S * ((alpha * Q + Lamda * tmp_ones) / tmp)

			#update theta
			tmp1 = 1/torch.trace(H1.T@(D1 - A1)@H1)
			tmp2 = 1/torch.trace(H2.T@(D2 - A2)@H2)
			tmp3 = 1/torch.trace(H3.T@(D3 - A3)@H3)
			tmp = tmp1 + tmp2 + tmp3
			theta1 = tmp1 / tmp
			theta2 = tmp2 / tmp
			theta3 = tmp3 / tmp

			

			#if stop_rule == 2:
			obj = compute_obj_3(X1,X2,X3,S,W1,W2, W3, H1,H2,H3, D1,A1,D2,A2,D3, A3,alpha,gamma,theta1,theta2, theta3)
			objs[i,0] = obj
			error = torch.abs(obj_old - obj) / obj_old
			if  (error < 1e-5 and i > 0) or i == Maxiter - 1:
				print('number of epoch:', i+1)
				print('obj:',obj)
				print('converged!')
				break

			obj_old = obj

			print('number of epoch:', i+1)
			print('obj:',obj)

		S = (S + S.T)/2
		use_epochs = i+1

		return W1,W2,W3, H1,H2,H3,S,use_epochs,objs


	def run(self):
		'''
		Run the JSNMF model. Init time and main function time are recorded.

		Returns
		----------
		result  :  dict storing some information during the model running

		'''
		start = time.time()
		alpha, gamma, Inits = self.parameter_selection()
		end = time.time()
		self.init_t = end - start
		self.alpha = alpha
		self.gamma = gamma
		self.Inits = Inits

		print('Init done')

		start = time.time()
		W1,W2,W3, H1,H2,H3,S,use_epochs,objs = self.gcnmf()
		end = time.time()
		self.run_t = end - start
		self.used_epoch = use_epochs

		result = dict(W1 = W1, W2 = W2, W3 = W3,
					H1 = H1, H2 = H2,
					H3= H3, 
					S = S, used_epoch = used_epoch,
					objs = objs,
					init_t = self.init_t,
					run_t  = self.run_t)

		self.result = result
		#return result

	def cluster(self, K = 50, step = 0.01, start = 2.7, upper = 4 ,seed = 3):

		'''
		Use louvain to cluster the cells based on the complete graph S, note that
		the function tries to find  a partition that has the same number of clusters
		as the true labels, the resolution parameter of louvain is found using binary search

		Parameters
		----------
		K      : (0, N) int, parameter of Wtrim
			     Number of neighbors to retain
		step   :  the step of binary search to find the partition, default 0.01
		start  :  start searching point of binary search
		upper  :  the upper bound of the reolution paramter to be searched
		seed   : seed parameter for louvain algorithm

		Returns
		-------
		res_clu : resulting cluster labels for each cell

		Note that sometimes exact number of clusters as true labels may not be found, paramters
		need to be adjusted then, like step, seed and upper

		'''
		S = self.result['S']
		A = Wtrim(S, K = 50)
		A = A.numpy()

		num_c = self.num_c

		tmp_gamma = start
		#use louvain to cluster based on A
		clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma, seed = seed)
		tmp_c = len(np.unique(clusters))
		tmp_clu = clusters

		res_clu = None


		# use binary search to find the corret gamma parameter
		while True:
			if tmp_c == num_c:
				res_clu = tmp_clu
				break

			if tmp_c < num_c:
				tmp_gamma = tmp_gamma + step
			else:
				tmp_gamma = tmp_gamma - step

			if tmp_gamma < 0 or tmp_gamma > upper:
				break
			tmp_gamma = round(tmp_gamma,2)
			#print(tmp_res)
			clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma,seed = seed)
			tmp_c = len(np.unique(clusters))
			tmp_clu = clusters

		return res_clu

	def visualize(self, label, tag = False, **kwarg):

		'''
		Visualize based on the complete graph S using Umap

		Parameters
		----------
		label     : array, true or clustered (louvain result) labels for each cell
		tag		  : if recalculte umap embedding
		**kwarg   : kwarg for the umap 	    

		Returns
		-------
		res_clu : resulting cluster labels for each cell

		'''


		# transfer S to distance matrix first
		S = self.result['S']
		data = 1-S
		data = data-np.diag(data.diagonal())
		reducer = umap.UMAP(**kwarg)

		# avoid recompute umap embedding
		if self.embedding is None:
			#min_dist = 0.68, n_neighbors=12
			embedding = reducer.fit_transform(data)
			self.embedding = embedding

		# recaculate embedding if needed
		if tag is True:
			embedding = reducer.fit_transform(data)
			self.embedding = embedding


		# plt.figure(figsize=(3, 1.5), dpi=300)
		# visualize
		for i in range(1,label.max() + 1):
			ind = label == i
			rgb = (random.random(), random.random() /2, random.random()/2)
			plt.scatter(self.embedding[ind, 0],self.embedding[ind, 1], s = 1.5, label = i,color = rgb)
		plt.legend(ncol=2,bbox_to_anchor=(1, 1.2))
		plt.show()



class JSNMF_same_H:
	def __init__(self,
				 X1: torch.tensor,
				 X2: torch.tensor,
				 label,
				 max_epochs: int = 200):

		try: 
			self.X1 = X1.cuda()
			self.X2 = X2.cuda()
			
		except:
			self.X1 = X1
			self.X2 = X2
			
		
		self.label = label

		self.max_epochs = max_epochs
		self.num_c  = num_c = len(np.unique(self.label))
		self.embedding = None


	def parameter_selection(self):



		Inits = {}
		num_c = self.num_c

		W1, H1 =  nndsvd(self.X1,num_c)
		print('nnsvd W1 done')
		W2, H2 = nndsvd(self.X2,num_c)
		print('nnsvd W2 done')

		Inits['W1'] = W1
		Inits['W2'] = W2
		Inits['H1'] = H1
		Inits['H2'] = H2

		D1 = dist2(self.X1.T,self.X1.T)
		print('D1 done')
		S1 = affinityMatrix(D1,20) # cell by cell
		print('S1 done')
		D2 = dist2(self.X2.T,self.X2.T)
		print('D2 done')
		S2 = affinityMatrix(D2,20) #cell by cell
		print('S2 done')


		
		Inits['A1'] = S1
		Inits['D1'] = torch.diag(S1.sum(0))
		Inits['L1'] = Inits['D1'] - Inits['A1']

		Inits['A2'] = S2
		Inits['D2'] = torch.diag(S2.sum(0))
		Inits['L2'] = Inits['D2'] - Inits['A2']


		err1 = torch.square(torch.norm(self.X1 - W1@H1))
		err2 = torch.square(torch.norm(self.X2 - W2@H2))
		err3 = (H1@Inits['L1']@H1.T).trace()
		gamma = torch.abs((err1+err2)/err3)
		gamma = gamma/5

		return gamma,Inits 

	def gcnmf(self):
		'''
		Main function to run the model. 

		Returns
		----------
        W1,W2,H1,H2    : resulting values of init  after running the model
		S			   : resulting complete graph

		use_epochs     : used epochs to converge

		objs		   : value of objective during each iteration

		'''
		Maxiter = self.max_epochs
		X1 = self.X1
		X2 = self.X2
		gamma = self.gamma
		W1 = self.Inits['W1']
		W2 = self.Inits['W2']
		H = self.Inits['H1'].T

		A1 = self.Inits['A1']
		D1 = self.Inits['D1']
		A2 = self.Inits['A2']
		D2 = self.Inits['D2']


		try:
			theta1 = torch.tensor(1/2).cuda()
			theta2 = torch.tensor(1/2).cuda()
			yita = torch.tensor(1.).cuda()
			obj_old = torch.tensor(1.).cuda()
			objs = torch.zeros((Maxiter,1)).cuda()
		except:
			theta1 = torch.tensor(1/2)
			theta2 = torch.tensor(1/2)
			yita = torch.tensor(1.).cuda()
			obj_old = torch.tensor(1.)
			objs = torch.zeros((Maxiter,1))

		for i in range(Maxiter):

			# update W1 W2 using bpp algorithm
			# W1, _ = nnlsm_blockpivot(H1, X1.T, False, W1.T) 
			# W2, _ = nnlsm_blockpivot(H2, X2.T, False, W2.T) 
			# W1 = W1.T
			# W2 = W2.T

			# updating w1 w2 via multiplication rule
			HtH = H.T @ H
			tmp1 = W1@HtH
			tmp1[tmp1 < 1e-10] = 1e-10
			W1 = W1 * (X1@H) /tmp1
			tmp2 = W2@HtH
			tmp2[tmp2 < 1e-10] = 1e-10
			W2 = W2 * (X2@H) / tmp2

			#update H
			W1tW1 = W1.T @ W1
			W2tW2 = W2.T @ W2
			HtH = H.T @ H

			# H = H.*(X1'*W1+X2'*W2+yita*H+gamma*(theta1.^2*A1+theta2.^2*A2)*H)./max(H*(W1tW1+W2tW2)+yita*H*HtH+gamma*(theta1.^2*D1+theta2.^2*D2)*H,1e-10);
			# max(H*(W1tW1+W2tW2)+yita*H*HtH+gamma*(theta1.^2*D1+theta2.^2*D2)*H,1e-10)
			res = H @ (W1tW1+W2tW2) + yita*H@HtH + gamma*(  theta1.square() * D1  + theta2.square() *D2  )@H
			res[res<1e-10] = 1e-10
			H = H * (X1.T @W1 + X2.T @W2 + yita*H + gamma*(theta1.square() * A1 + theta2.square() * A2)@H   / res)


			#update theta
			tmp1 = 1/ (H.T@(D1-A1)@H).trace()
			tmp2 = 1/ (H.T@(D2-A2)@H).trace()
			tmp = tmp1 + tmp2
			theta1 = tmp1/tmp
			theta2 = tmp2/tmp


			#if stop_rule == 2:
			obj = compute_obj_H(X1,X2,W1,W2,H,D1,A1,D2,A2,gamma,theta1,theta2)
			objs[i,0] = obj
			error = torch.abs(obj_old - obj) / obj_old
			if  (error < 1e-5 and i > 0) or i == Maxiter - 1:
				print('number of epoch:', i+1)
				print('obj:',obj)
				print('converged!')
				break

			obj_old = obj

			print('number of epoch:', i+1)
			print('obj:',obj)


		use_epochs = i+1

		return W1,W2,H,use_epochs,objs
	

	def run(self):
		'''
		Run the JSNMF model. Init time and main function time are recorded.

		Returns
		----------
		result  :  dict storing some information during the model running

		'''
		start = time.time()
		alpha, gamma, Inits = self.parameter_selection()
		end = time.time()
		self.init_t = end - start
		self.alpha = alpha
		self.gamma = gamma
		self.Inits = Inits

		print('Init done')

		start = time.time()
		W1,W2,H,used_epoch,objs  = self.gcnmf()
		end = time.time()
		self.run_t = end - start
		self.used_epoch = used_epoch

		result = dict(W1 = W1, W2 = W2,
					H = H,
					used_epoch = used_epoch,
					objs = objs,
					init_t = self.init_t,
					run_t  = self.run_t)

		self.result = result


	def cluster(self, K = 50, step = 0.01, start = 2.7, upper = 4 ,seed = 3):

		'''
		Use louvain to cluster the cells based on the complete graph S, note that
		the function tries to find  a partition that has the same number of clusters
		as the true labels, the resolution parameter of louvain is found using binary search

		Parameters
		----------
		K      : (0, N) int, parameter of Wtrim
			     Number of neighbors to retain
		step   :  the step of binary search to find the partition, default 0.01
		start  :  start searching point of binary search
		upper  :  the upper bound of the reolution paramter to be searched
		seed   : seed parameter for louvain algorithm

		Returns
		-------
		res_clu : resulting cluster labels for each cell

		Note that sometimes exact number of clusters as true labels may not be found, paramters
		need to be adjusted then, like step, seed and upper

		'''
		H = self.result['H']
		A = cos_opt(H)
		A = Wtrim(A,K)
		A = A.numpy()

		num_c = self.num_c

		tmp_gamma = start
		#use louvain to cluster based on A
		clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma, seed = seed)
		tmp_c = len(np.unique(clusters))
		tmp_clu = clusters

		res_clu = None


		# use binary search to find the corret gamma parameter
		while True:
			if tmp_c == num_c:
				res_clu = tmp_clu
				break

			if tmp_c < num_c:
				tmp_gamma = tmp_gamma + step
			else:
				tmp_gamma = tmp_gamma - step

			if tmp_gamma < 0 or tmp_gamma > upper:
				break
			tmp_gamma = round(tmp_gamma,2)
			#print(tmp_res)
			clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma,seed = seed)
			tmp_c = len(np.unique(clusters))
			tmp_clu = clusters

		return res_clu




class JSNMF_be5:
	def __init__(self,
				 X11, X12, X21, X23,X31,X34,X41, X45,X51,X56,
				 num_c: list,
				 max_epochs: int = 500):
		
		self.X11 = X11
		self.X12 = X12

	
		try:
			self.X11 = X11.cuda()
			self.X12 = X12.cuda()
			self.X21 = X21.cuda()
			self.X23 = X23.cuda()
			self.X31 = X31.cuda()
			self.X34 = X34.cuda()
			self.X41 = X41.cuda()
			self.X45 = X45.cuda()
			self.X51 = X51.cuda()
			self.X56 = X56.cuda()
			
		except:
			self.X11 = X11
			self.X12 = X12
			self.X21 = X21
			self.X23 = X23
			self.X31 = X31
			self.X34 = X34
			self.X41 = X41
			self.X45 = X45
			self.X51 = X51
			self.X56 = X56
						# for i in range(self.n_mod):
			# 	self.X1.append(X1[i])
			# 	self.X2.append(X2[i])
		
		self.num_c = num_c

		self.max_epochs = max_epochs
		#self.num_c  = num_c = len(np.unique(self.label))
		self.embedding = None



	def parameter_selection(self):
		'''
		Calculates the initilization values for the model. 

		Returns
		----------
		alpha   : init hyperparameter alpha
		beta    : init hyperparameter beta
		Inits   : dict stroing the init values

		'''
		Inits = {}

		num_c = self.num_c
		W11, H11 =  nndsvd(self.X11,num_c[0])
		W21, H21 =  nndsvd(self.X21,num_c[1])
		W31, H31 =  nndsvd(self.X31,num_c[2])
		W41, H41 =  nndsvd(self.X41,num_c[3])
		W51, H51 =  nndsvd(self.X51,num_c[4])



		W2, H12 =  nndsvd(self.X12,num_c[0])
		W3, H23 =  nndsvd(self.X23,num_c[1])
		W4, H34 =  nndsvd(self.X34,num_c[2])
		W5, H45 =  nndsvd(self.X45,num_c[3])
		W6, H56 =  nndsvd(self.X56,num_c[4])

		Inits['W1'] = (W11+W21+W31+W41+W51)/5
		Inits['W2'] = W2
		Inits['W3'] = W3
		Inits['W4'] = W4
		Inits['W5'] = W5
		Inits['W6'] = W6

		Inits['H11'] = H11
		Inits['H12'] = H12
		Inits['H21'] = H21
		Inits['H23'] = H23
		Inits['H31'] = H31
		Inits['H34'] = H34
		Inits['H41'] = H41
		Inits['H45'] = H45
		Inits['H51'] = H51
		Inits['H56'] = H56

		D11 = dist2(self.X11.T,self.X11.T)
		S11 = affinityMatrix(D11,20)
		D12 = dist2(self.X12.T,self.X12.T)
		S12 = affinityMatrix(D12,20)

		D21 = dist2(self.X21.T,self.X21.T)
		S21 = affinityMatrix(D21,20)
		D23 = dist2(self.X23.T,self.X23.T)
		S23 = affinityMatrix(D23,20)

		D31 = dist2(self.X31.T,self.X31.T)
		S31 = affinityMatrix(D31,20)
		D34 = dist2(self.X34.T,self.X34.T)
		S34 = affinityMatrix(D34,20)

		D41 = dist2(self.X41.T,self.X41.T)
		S41 = affinityMatrix(D41,20)
		D45 = dist2(self.X45.T,self.X45.T)
		S45 = affinityMatrix(D45,20)

		D51 = dist2(self.X51.T,self.X51.T)
		S51 = affinityMatrix(D51,20)
		D56 = dist2(self.X56.T,self.X56.T)
		S56 = affinityMatrix(D56,20)

		Inits['A11']= S11
		Inits['D11']= torch.diag(S11.sum(0))
		Inits['L11'] = Inits['D11'] - Inits['A11']
		Inits['A12']= S12
		Inits['D12']= torch.diag(S12.sum(0))
		Inits['L12'] = Inits['D12'] - Inits['A12']

		Inits['A21']= S21
		Inits['D21']= torch.diag(S21.sum(0))
		Inits['L21'] = Inits['D21'] - Inits['A21']
		Inits['A23']= S23
		Inits['D23']= torch.diag(S23.sum(0))
		Inits['L23'] = Inits['D23'] - Inits['A23']

		Inits['A31']= S31
		Inits['D31']= torch.diag(S31.sum(0))
		Inits['L31'] = Inits['D31'] - Inits['A31']
		Inits['A34']= S34
		Inits['D34']= torch.diag(S34.sum(0))
		Inits['L34'] = Inits['D34'] - Inits['A34']

		Inits['A41']= S41
		Inits['D41']= torch.diag(S41.sum(0))
		Inits['L41'] = Inits['D41'] - Inits['A41']
		Inits['A45']= S45
		Inits['D45']= torch.diag(S45.sum(0))
		Inits['L45'] = Inits['D45'] - Inits['A45']

		Inits['A51']= S51
		Inits['D51']= torch.diag(S51.sum(0))
		Inits['L51'] = Inits['D51'] - Inits['A51']
		Inits['A56']= S56
		Inits['D56']= torch.diag(S56.sum(0))
		Inits['L56'] = Inits['D56'] - Inits['A56']


		snf_W1 = SNF(S11,S12,20)
		snf_W2 = SNF(S21,S23,20)
		snf_W3 = SNF(S31,S34,20)
		snf_W4 = SNF(S41,S45,20)
		snf_W5 = SNF(S51,S56,20)

		Inits['S'] = []
		Inits['S'].append(snf_W1)
		Inits['S'].append(snf_W2)
		Inits['S'].append(snf_W3)
		Inits['S'].append(snf_W4)
		Inits['S'].append(snf_W5)


		H11tH11 = H11.T @ H11
		H12tH12 = H12.T @ H12
		H21tH21 = H21.T @ H21
		H23tH23 = H23.T @ H23		
		H31tH31 = H31.T @ H31
		H34tH34 = H34.T @ H34
		H41tH41 = H41.T @ H41
		H45tH45 = H45.T @ H45
		H51tH51 = H51.T @ H51
		H56tH56 = H56.T @ H56

		err11 = torch.square(torch.norm(self.X11 - W11@H11))
		err12 = torch.square(torch.norm(self.X12 - W2@H12))

		err21 = torch.square(torch.norm(self.X21 - W21@H21))
		err23 = torch.square(torch.norm(self.X23 - W3@H23))


		err31 = torch.square(torch.norm(self.X31 - W31@H31))
		err34 = torch.square(torch.norm(self.X34 - W4@H34))


		err41 = torch.square(torch.norm(self.X41 - W41@H41))
		err45 = torch.square(torch.norm(self.X45 - W5@H45))


		err51 = torch.square(torch.norm(self.X51 - W51@H51))
		err56 = torch.square(torch.norm(self.X56 - W6@H56))


		errS1 =  0.5 * ( torch.square(torch.norm(Inits['S'][0] - H11tH11)) +  torch.square(torch.norm(Inits['S'][0] - H12tH12)))
		errS2 =  0.5 * ( torch.square(torch.norm(Inits['S'][1] - H21tH21)) +  torch.square(torch.norm(Inits['S'][1] - H23tH23)))
		errS3 =  0.5 * ( torch.square(torch.norm(Inits['S'][2] - H31tH31)) +  torch.square(torch.norm(Inits['S'][2] - H34tH34)))
		errS4 =  0.5 * ( torch.square(torch.norm(Inits['S'][3] - H41tH41)) +  torch.square(torch.norm(Inits['S'][3] - H45tH45)))
		errS5 =  0.5 * ( torch.square(torch.norm(Inits['S'][4] - H51tH51)) +  torch.square(torch.norm(Inits['S'][4] - H56tH56)))

		err7 = 0.25 * (H11@Inits['L11']@H11.T + H12@Inits['L12']@H12.T + H21@Inits['L21']@H21.T + H23@Inits['L23']@H23.T).trace()
		err8 = 0.25 * (H31@Inits['L31']@H31.T + H34@Inits['L34']@H34.T + H41@Inits['L41']@H41.T + H45@Inits['L45']@H45.T).trace()
		err9 =  0.25 * (H51@Inits['L51']@H51.T  + H56@Inits['L56']@H56.T ).trace()
		tot1 = err11+err12+err21+err23+err31+err34+err41+err45+err51+err56
		alpha = tot1/(errS1+errS2+errS3+errS4+errS5)
		gamma = torch.abs(tot1/(err7+err8+err9))

		alpha = alpha / 10
		gamma = gamma /10

		return alpha,gamma, Inits


	def gcnmf(self):
			'''
			Main function to run the model. 

			Returns
			----------
			W1,W2,H1,H2    : resulting values of init  after running the model
			S			   : resulting complete graph

			use_epochs     : used epochs to converge

			objs		   : value of objective during each iteration

			'''
			Maxiter = self.max_epochs
			W1 = self.Inits['W1']
			W2 = self.Inits['W2']
			W3 = self.Inits['W3']
			W4 = self.Inits['W4']
			W5 = self.Inits['W5']
			W6 = self.Inits['W6']
			gamma = self.gamma
			alpha = self.alpha

			H11 = self.Inits['H11'].T
			H12 = self.Inits['H12'].T
			H21 = self.Inits['H21'].T
			H23 = self.Inits['H23'].T
			A11 = self.Inits['A11']
			D11 = self.Inits['D11']
			A12 = self.Inits['A12']
			D12 = self.Inits['D12']
			A21 = self.Inits['A21']
			D21 = self.Inits['D21']
			A23 = self.Inits['A23']
			D23 = self.Inits['D23']
			S1 = self.Inits['S'][0]
			S1 = S1 / S1.sum(dim =0,keepdim = True)
			n1 = S1.size(0)
			S2 = self.Inits['S'][1]
			S2 = S2 / S2.sum(dim =0,keepdim = True)
			n2 = S2.size(0)

			H31 = self.Inits['H31'].T
			H34 = self.Inits['H34'].T
			H41 = self.Inits['H41'].T
			H45 = self.Inits['H45'].T
			A31 = self.Inits['A31']
			D31 = self.Inits['D31']
			A34 = self.Inits['A34']
			D34 = self.Inits['D34']
			A41 = self.Inits['A41']
			D41 = self.Inits['D41']
			A45 = self.Inits['A45']
			D45 = self.Inits['D45']
			S3 = self.Inits['S'][2]
			S3 = S3 / S3.sum(dim =0,keepdim = True)
			n3 = S3.size(0)
			S4 = self.Inits['S'][3]
			S4 = S4 / S4.sum(dim =0,keepdim = True)
			n4 = S4.size(0)



			H51 = self.Inits['H51'].T
			H56 = self.Inits['H56'].T
			A51 = self.Inits['A51']
			D51 = self.Inits['D51']
			A56 = self.Inits['A56']
			D56 = self.Inits['D56']
			S5 = self.Inits['S'][4]
			S5 = S5 / S5.sum(dim =0,keepdim = True)
			n5 = S5.size(0)


			theta11 = 1/10
			theta12 = 1/10
			theta21 = 1/10
			theta23 = 1/10
			theta31 = 1/10
			theta34 = 1/10
			theta41 = 1/10
			theta45 = 1/10
			theta51 = 1/10
			theta56 = 1/10

			obj_old = 1
			yita11 = 1
			yita12 = 0.5
			yita21 = 1 
			yita23 = 0.5
			lamda = 0.5
			yita31 = 1
			yita34 = 0.5
			yita41 = 1
			yita45 = 0.5
			yita51 = 1
			yita56 = 0.5
			objs = torch.zeros((Maxiter,1))

			for i in range(Maxiter):
				H11tH11 = H11.T @ H11
				H21tH21 = H21.T @ H21
				H31tH31 = H31.T @ H31
				H41tH41 = H41.T @ H41
				H51tH51 = H51.T @ H51
				W1 = W1*(self.X11 @ H11   +  self.X21 @ H21 + self.X31 @ H31 + self.X41 @ H41 + self.X51 @ H51)
				tmp = W1 @ (H11tH11 + H21tH21 + H31tH31 + H41tH41 + H51tH51)
				tmp[tmp<1e-10] = 1e-10
				W1 = W1 / tmp
				W2, _ = nnlsm_blockpivot(H12, self.X12.T, False, W2.T) 
				W3, _ = nnlsm_blockpivot(H23, self.X23.T, False, W3.T) 
				W4, _ = nnlsm_blockpivot(H34, self.X34.T, False, W4.T) 
				W5, _ = nnlsm_blockpivot(H45, self.X45.T, False, W5.T) 
				W6, _ = nnlsm_blockpivot(H56, self.X56.T, False, W6.T) 
				W2 = W2.T
				W3 = W3.T
				W4 = W4.T
				W5 = W5.T
				W6 = W6.T


				#update H1,H2
				W1tW1 = W1.T@W1
				W2tW2 = W2.T@W2
				W3tW3 = W3.T@W3
				H12tH12 = H12.T@H12
				H23tH23 = H23.T@H23
				H11 = H11 * (alpha * S1.T@H11 + self.X11.T@W1 + gamma* (theta11**2)*A11@H11+yita11*H11)
				tmp = gamma*(theta11**2)*D11@H11+(alpha+yita11)*H11@H11tH11+H11@W1tW1
				tmp[tmp<1e-10] = 1e-10
				H11 = H11 / tmp

				H12 = H12 * (alpha * S1.T@H12 + self.X12.T@W2 + gamma* (theta12**2)*A12@H12+yita12*H12)
				tmp = gamma*(theta12**2)*D12@H12+(alpha+yita12)*H12@H12tH12+H12@W2tW2
				tmp[tmp<1e-10] = 1e-10
				H12 = H12 / tmp

				H21 = H21 * (alpha * S2.T@H21 + self.X21.T@W1 + gamma* (theta21**2)*A21@H21+yita21*H21)
				tmp = gamma*(theta21**2)*D21@H21+(alpha+yita21)*H21@H21tH21+H21@W1tW1
				tmp[tmp<1e-10] = 1e-10
				H21 = H21 / tmp

				H23 = H23 * (alpha * S2.T@H23 + self.X23.T@W3 + gamma* (theta23**2)*A23@H23+yita23*H23)
				tmp = gamma*(theta23**2)*D23@H23+(alpha+yita23)*H23@H23tH23+H23@W3tW3
				tmp[tmp<1e-10] = 1e-10
				H23 = H23 / tmp


				W4tW4 = W4.T@W4
				W5tW5 = W5.T@W5
				W6tW6 = W6.T@W6
				H34tH34 = H34.T@H34
				H45tH45 = H45.T@H45
				H56tH56 = H56.T@H56
				H31 = H31 * (alpha * S3.T@H31 + self.X31.T@W1 + gamma* (theta31**2)*A31@H31+yita31*H31)
				tmp = gamma*(theta31**2)*D31@H31+(alpha+yita31)*H31@H31tH31+H31@W1tW1
				tmp[tmp<1e-10] = 1e-10
				H31 = H31 / tmp

				H34 = H34 * (alpha * S3.T@H34 + self.X34.T@W4 + gamma* (theta34**2)*A34@H34+yita34*H34)
				tmp = gamma*(theta34**2)*D34@H34+(alpha+yita34)*H34@H34tH34+H34@W4tW4
				tmp[tmp<1e-10] = 1e-10
				H34 = H34 / tmp

				H41 = H41 * (alpha * S4.T@H41 + self.X41.T@W1 + gamma* (theta41**2)*A41@H41+yita41*H41)
				tmp = gamma*(theta41**2)*D41@H41+(alpha+yita41)*H41@H41tH41+H41@W1tW1
				tmp[tmp<1e-10] = 1e-10
				H41 = H41 / tmp

				H45 = H45 * (alpha * S4.T@H45 + self.X45.T@W5 + gamma* (theta45**2)*A45@H45+yita45*H45)
				tmp = gamma*(theta45**2)*D45@H45+(alpha+yita45)*H45@H45tH45+H45@W5tW5
				tmp[tmp<1e-10] = 1e-10
				H45 = H45 / tmp


				H51 = H51 * (alpha * S5.T@H51 + self.X51.T@W1 + gamma* (theta51**2)*A51@H51+yita51*H51)
				tmp = gamma*(theta51**2)*D51@H51+(alpha+yita51)*H51@H51tH51+H51@W1tW1
				tmp[tmp<1e-10] = 1e-10
				H51 = H51 / tmp

				H56 = H45 * (alpha * S5.T@H56 + self.X56.T@W6 + gamma* (theta56**2)*A56@H56+yita56*H56)
				tmp = gamma*(theta56**2)*D56@H56+(alpha+yita56)*H56@H56tH56+H56@W6tW6
				tmp[tmp<1e-10] = 1e-10
				H56 = H56 / tmp


				# update S

				H11tH11 = H11.T @ H11
				H12tH12 = H12.T @ H12
				H21tH21 = H21.T @ H21
				H23tH23 = H23.T @ H23
				Q1 = 0.5 * (H11tH11 + H12tH12)
				Q2 = 0.5 * (H21tH21 + H23tH23)

				H31tH31 = H31.T @ H31
				H34tH34 = H34.T @ H34
				H41tH41 = H41.T @ H41
				H45tH45 = H45.T @ H45
				H51tH51 = H51.T @ H51
				H56tH56 = H56.T @ H56
				Q3 = 0.5 * (H31tH31 + H34tH34)
				Q4 = 0.5 * (H41tH41 + H45tH45)
				Q5 = 0.5 * (H51tH51 + H56tH56)


				# fill in some here

				tmp = alpha*S1 + lamda * S1.sum(dim = 0, keepdim =True)
				tmp[tmp<1e-10] = 1e-10
				try: 
					tmp_ones = torch.ones((n1,n1)).cuda()
				except:
					tmp_ones = torch.ones((n1,n1))
				S1 = S1 * ((alpha * Q1 + lamda * tmp_ones) / tmp)


				
				tmp = alpha*S2 + lamda * S2.sum(dim = 0, keepdim =True)
				tmp[tmp<1e-10] = 1e-10
				try: 
					tmp_ones = torch.ones((n2,n2)).cuda()
				except:
					tmp_ones = torch.ones((n2,n2))
				S2 = S2 * ((alpha * Q2 + lamda * tmp_ones) / tmp)

				
				tmp = alpha*S3 + lamda * S3.sum(dim = 0, keepdim =True)
				tmp[tmp<1e-10] = 1e-10
				try: 
					tmp_ones = torch.ones((n3,n3)).cuda()
				except:
					tmp_ones = torch.ones((n3,n3))
				S3 = S3 * ((alpha * Q3 + lamda * tmp_ones) / tmp)

				
				tmp = alpha*S4 + lamda * S4.sum(dim = 0, keepdim =True)
				tmp[tmp<1e-10] = 1e-10
				try: 
					tmp_ones = torch.ones((n4,n4)).cuda()
				except:
					tmp_ones = torch.ones((n4,n4))
				S4 = S4 * ((alpha * Q4 + lamda * tmp_ones) / tmp)

				
				tmp = alpha*S5 + lamda * S5.sum(dim = 0, keepdim =True)
				tmp[tmp<1e-10] = 1e-10
				try: 
					tmp_ones = torch.ones((n5,n5)).cuda()
				except:
					tmp_ones = torch.ones((n5,n5))
				S5 = S5 * ((alpha * Q5 + lamda * tmp_ones) / tmp)



				#update theta

				tmp11 = 1/torch.trace(H11.T@(D11-A11)@H11)
				tmp12 = 1/torch.trace(H12.T@(D12-A12)@H12)
				tmp21 = 1/torch.trace(H21.T@(D21-A21)@H21)
				tmp23 = 1/torch.trace(H23.T@(D23-A23)@H23)
				tmp31 = 1/torch.trace(H31.T@(D31-A31)@H31)
				tmp34 = 1/torch.trace(H34.T@(D34-A34)@H34)
				tmp41 = 1/torch.trace(H41.T@(D41-A41)@H41)
				tmp45 = 1/torch.trace(H45.T@(D45-A45)@H45)
				tmp51 = 1/torch.trace(H51.T@(D51-A51)@H51)
				tmp56 = 1/torch.trace(H56.T@(D56-A56)@H56)

				tmp = (tmp11+tmp12)+(tmp21+tmp23)+(tmp31+tmp34)+(tmp41+tmp45)+(tmp51+tmp56)
				theta11 = tmp11/tmp
				theta12 = tmp12/tmp
				theta21 = tmp21/tmp
				theta23 = tmp23/tmp
				theta31 = tmp31/tmp 
				theta34 = tmp34/tmp
				theta41 = tmp41/tmp
				theta45 = tmp45/tmp
				theta51 = tmp51/tmp
				theta56 = tmp56/tmp
							

				obj = compute_obj_5(self.X11,self.X12,self.X21,self.X23,self.X31,self.X34,self.X41,self.X45,self.X51,self.X56,S1,S2,S3,S4,S5,W1,W2,W3,W4,W5,W6,H11,H12,H21,H23,H31,H34,H41,H45,H51,H56,D11,A11,D12,A12,D21,A21,D23,A23,D31,A31,D34,A34,D41,A41,D45,A45,D51,A51,D56,A56, alpha,gamma,theta11,theta12,theta21,theta23,theta31,theta34,theta41,theta45,theta51,theta56)
				objs[i,0] = obj
				error = torch.abs(obj_old - obj) / obj_old
				if  (error < 1e-5 and i > 0) or i == Maxiter - 1:
					print('number of epoch:', i+1)
					print('obj:',obj)
					print('converged!')
					break

				obj_old = obj

				print('number of epoch:', i+1)
				print('obj:',obj)
			
			S = []
			S.append((S1+S1.T)/2)
			S.append((S2+S2.T)/2)
			S.append((S3+S3.T)/2)
			S.append((S4+S4.T)/2)
			S.append((S5+S5.T)/2)

			use_epochs = i+1
			return W1,W2,W3,W4,W5,W6,H11,H12,H21,H23,H31,H34,H41,H45,H51,H56,S, use_epochs,objs
							
				
				
	def run(self):
		'''
		Run the JSNMF model. Init time and main function time are recorded.

		Returns
		----------
		result  :  dict storing some information during the model running

		'''
		start = time.time()
		alpha, gamma, Inits = self.parameter_selection()
		end = time.time()
		self.init_t = end - start
		self.alpha = alpha
		self.gamma = gamma
		self.Inits = Inits

		print('Init done')

		start = time.time()
		W1,W2,W3,W4,W5,W6,H11,H12,H21,H23,H31,H34,H41,H45,H51,H56,S, use_epochs,objs = self.gcnmf()
		end = time.time()
		self.run_t = end - start
		self.used_epoch = use_epochs

		result = dict(W1 = W1, W2 = W2, W3 = W3,
					W4 = W4, W5 = W5,W6 =W6,
					H11 = H11, H12 = H12,
					H21 = H21, H23 = H23,
					H31 = H31, H34 = H34,
					H41 = H41, H45 = H45,
					S = S, used_epoch = used_epoch,
					objs = objs,
					init_t = self.init_t,
					run_t  = self.run_t)

		self.result = result

	def cluster(self,i, K = 50, step = 0.01, start = 2.7, upper = 4 ,seed = 3):

		'''
		Use louvain to cluster the cells based on the complete graph S, note that
		the function tries to find  a partition that has the same number of clusters
		as the true labels, the resolution parameter of louvain is found using binary search

		Parameters
		----------
		i	   : the modality to cluster, index starting from 0
		K      : (0, N) int, parameter of Wtrim
			     Number of neighbors to retain
		step   :  the step of binary search to find the partition, default 0.01
		start  :  start searching point of binary search
		upper  :  the upper bound of the reolution paramter to be searched
		seed   : seed parameter for louvain algorithm

		Returns
		-------
		res_clu : resulting cluster labels for each cell

		Note that sometimes exact number of clusters as true labels may not be found, paramters
		need to be adjusted then, like step, seed and upper

		'''
		S = self.result['S'][i]
		A = Wtrim(S, K = 50)
		A = A.numpy()

		num_c = self.num_c[i]

		tmp_gamma = start
		#use louvain to cluster based on A
		clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma, seed = seed)
		tmp_c = len(np.unique(clusters))
		tmp_clu = clusters

		res_clu = None


		# use binary search to find the corret gamma parameter
		while True:
			if tmp_c == num_c:
				res_clu = tmp_clu
				break

			if tmp_c < num_c:
				tmp_gamma = tmp_gamma + step
			else:
				tmp_gamma = tmp_gamma - step

			if tmp_gamma < 0 or tmp_gamma > upper:
				break
			tmp_gamma = round(tmp_gamma,2)
			#print(tmp_res)
			clusters, q_stat = bct.community_louvain(A,gamma = tmp_gamma,seed = seed)
			tmp_c = len(np.unique(clusters))
			tmp_clu = clusters

		return res_clu

	def visualize(self, label, tag = False, **kwarg):

		'''
		Visualize based on the complete graph S using Umap

		Parameters
		----------
		label     : array, true or clustered (louvain result) labels for each cell
		tag		  : if recalculte umap embedding
		**kwarg   : kwarg for the umap 	    

		Returns
		-------
		res_clu : resulting cluster labels for each cell

		'''


		# transfer S to distance matrix first
		H11 =self.result['H11'].numpy()
		H21 =self.result['H21'].numpy()
		H31 =self.result['H31'].numpy()
		H41 =self.result['H41'].numpy()
		H51 =self.result['H51'].numpy()
		data = np.vstack((H11,H21,H31,H41,H51))
		reducer = umap.UMAP(**kwarg)

		# avoid recompute umap embedding
		if self.embedding is None:
			#min_dist = 0.68, n_neighbors=12
			embedding = reducer.fit_transform(data)
			self.embedding = embedding

		# recaculate embedding if needed
		if tag is True:
			embedding = reducer.fit_transform(data)
			self.embedding = embedding


		# plt.figure(figsize=(3, 1.5), dpi=300)
		# visualize
		# for i in range(1,label.max() + 1):
		# 	ind = label == i
		# 	rgb = (random.random(), random.random() /2, random.random()/2)
		# 	plt.scatter(self.embedding[ind, 0],self.embedding[ind, 1], s = 1.5, label = i,color = rgb)
		# plt.legend(ncol=2,bbox_to_anchor=(1, 1.2))
		# plt.show()
				


			





		



