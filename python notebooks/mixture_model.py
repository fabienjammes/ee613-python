from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn

import numpy as np
from numpy import dot
from numpy.linalg import inv
from numpy.linalg import pinv

import time
from utils import plot_GMM

class GMM():
    def __init__(self, D = 1, K = 2,  reg_factor = 1e-6):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        self.reg_factor =  reg_factor 

        
    def init_kmeans(self):
        kMM = KMeans(n_clusters=self.K).fit(self.x)
        self.means_ = kMM.cluster_centers_
        for i in range(self.K):
            self.covariances_[i] = np.cov(self.x[kMM.labels_==i].T) + np.eye(self.D)*self.reg_factor
        
    def init_random(self):
        self.means_ = self.x[np.random.choice(len(self.x),size = self.K)]
        for i in range(self.K):
            self.covariances_[i] = np.cov(self.x.T)

    def fit(self,x, max_iter = 10, init_type = 'kmeans', threshold = 1e-4, n_init = 5):
        self.x = x
        self.N = len(self.x) #number of datapoints
        self.Ls = np.zeros((self.N,self.K)) #posterior probability of z
        self.zs = np.zeros((self.N,self.K)) #posterior probability of z

        self.threshold = threshold
        
        best_params = ()
        Lmax = -np.inf
        for it in range(n_init):
            if init_type == 'kmeans':
                self.init_kmeans()
            elif init_type == 'random':
                self.init_random()

            for i in range(max_iter):
                print('Iteration {}'.format(i))
                self.expectation()
                self.maximization()
                print(self.L)
                if np.abs(self.prev_L-self.L) < self.threshold:
                    break
                    
            if self.L > Lmax:
                Lmax = self.L
                best_params = (self.L, self.weights_.copy(), self.means_.copy(), self.covariances_.copy(), self.zs.copy(), self.Ns.copy())
            
        #return the best result
        self.L = Lmax
        self.weights_ = best_params[1]
        self.means_ = best_params[2]
        self.covariances_ = best_params[3]
        self.zs = best_params[4]
        self.Ns = best_params[5]
        print('Obtain best result with Log Likelihood: {}'.format(self.L))
        
    def expectation(self):
        for k in range(self.K):
            self.Ls[:,k] = self.weights_[k]*mvn.pdf(self.x,mean = self.means_[k], cov=self.covariances_[k])

        self.zs = self.Ls/np.sum(self.Ls,axis=1)[:,None] #normalize

        self.prev_L = self.L
        self.L = np.sum(np.log(np.sum(self.Ls, axis=1)))/self.N
        self.Ns = np.sum(self.zs,axis=0)
             
    def maximization(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 

            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]
            
            #update covariance
            x_reduce_mean = self.x-self.means_[k,:]
            #S_k = dot(x_reduce_mean.T, dot(np.diag(self.zs[:,k]), x_reduce_mean))
            sigma_k = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            
            self.covariances_[k,:] = sigma_k        
            
    def get_marginal(self,dim,dim_out=None):
        if dim_out is not None:
            means_, covariances_ = (self.means_[:,dim],self.covariances_[:,dim,dim_out])
        else:
            means_, covariances_ = (self.means_[:,dim],self.covariances_[:,dim,dim])
        return means_,covariances_
    
    def condition(self,x_in,dim_in,dim_out,h=None, return_gmm = False):
        mu_in, sigma_in = self.get_marginal(dim_in)
        mu_out, sigma_out = self.get_marginal(dim_out)
        _, sigma_in_out = self.get_marginal(dim=dim_in, dim_out = dim_out)
        
        if h is None:
            h = np.zeros(self.K)
            for k in range(self.K):
                h[k] = self.weights_[k]*mvn(mean=mu_in[k],cov=sigma_in[k]).pdf(x_in)
            h = h/np.sum(h) 
        
        #compute mu and sigma
        mu = []
        sigma = []
        for k in range(self.K):
            mu += [mu_out[k] + np.dot(sigma_in_out[k].T, np.dot(np.linalg.inv(sigma_in[k]), (x_in-mu_in[k]).T)).flatten()]
            sigma += [sigma_out[k] - np.dot(sigma_in_out[k].T, np.dot(np.linalg.inv(sigma_in[k]), sigma_in_out[k]))]
            
        mu,sigma = (np.asarray(mu),np.asarray(sigma))
        if return_gmm:
            return h,mu,sigma
        else:
            return self.moment_matching(h,mu,sigma)
        
    def moment_matching(self,h,mu,sigma):
        dim = mu.shape[1]
        sigma_out = np.zeros((dim, dim))
        mu_out = np.zeros(dim)
        for k in range(self.K):
            sigma_out += h[k]*(sigma[k] + np.outer(mu[k],mu[k]))
            mu_out += h[k]*mu[k]
            
        sigma_out -= np.outer(mu_out, mu_out)
        return mu_out,sigma_out
        
    def plot(self):
        fig,ax = plt.subplots()
        plot_GMM(self.means_, self.covariances_, ax)
        

class HDGMM(GMM):
    def __init__(self, D = 1, K = 2,  reg_factor = 1e-6, n_fac = 1):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        self.reg_factor =  reg_factor 
        self.n_fac = n_fac
             
    def maximization(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 

            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]
            
            #update covariance
            x_reduce_mean = self.x-self.means_[k,:]
            sigma_k = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            
            #modify the covariance by replacing the last (D-n_fac) eigen values by their average
            D,V = np.linalg.eig(sigma_k)
            sort_indexes = np.argsort(D)[::-1]
            D = np.concatenate([D[sort_indexes[:self.n_fac]], [np.mean(D[sort_indexes[self.n_fac:]])]*(self.D-self.n_fac)])
            V = V[:,sort_indexes]
            self.covariances_[k,:] = dot(V, dot(np.diag(D), V.T))+ np.eye(self.D)*self.reg_factor       
            
class semitiedGMM(GMM):
    def __init__(self, D = 1, K = 2,  reg_factor = 1e-6, bsf_param =  5E-2, n_step_variation = 50):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        self.reg_factor =  reg_factor 
        self.bsf_param = bsf_param
        self.n_step_variation = n_step_variation
        self.B = np.eye(self.D)*self.bsf_param
        self.S = np.array([np.eye(D) for i in range(K)])
        self.Sigma_diag = np.array([np.eye(D) for i in range(K)])
        #def init_semitiedGMM(self):
        #self.H_init = pinv(self.B) + np.eye(self.D)*self.reg_factor
        #self.Sigma_diag_init = np.array([np.eye(self.D) for i in range(K)])
        #for i in range(self.K):
        #    eig_vals, V = np.linalg.eig(self.covariances_[i])
        #    self.Sigma_diag_init[i] = np.diag(eig_vals)
            
    def maximization(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 

            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]
            
            #calculate the sample covariances
            x_reduce_mean = self.x-self.means_[k,:]
            self.S[k] = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            
        #calculate H and the covariance
        for it in range(self.n_step_variation):
            for k in range(self.K):
                self.Sigma_diag[k] = np.diag(np.diag(dot(self.B, dot(self.S[k], self.B.T))))
                
            for i in range(self.D):
                C = pinv(self.B.T)*np.linalg.det(self.B)
                G = np.zeros((self.D,self.D))
                for k in range(self.K):
                    G += dot(self.S[k], np.sum(self.zs[:,k]))/self.Sigma_diag[k, i, i]
                self.B[i] = dot(C[i],pinv(G))*np.sqrt(np.sum(self.zs)/dot(C[i], dot(pinv(G), C[i].T)))
        self.H = pinv(self.B) + np.eye(self.D)*self.reg_factor
        for k in range(self.K):
            self.covariances_[k,:] = dot(self.H, dot(self.Sigma_diag[k], self.H.T))   
        
        
class MFA(GMM):
    def __init__(self, D = 1, K = 2, n_fac = 1, reg_factor = 1e-6):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        

        
        self.Lambda_ = np.array([np.zeros((D,n_fac)) for i in range(K)])
        self.Psi_ = np.array([np.eye(D) for i in range(K)])
        self.n_fac = n_fac
        self.reg_factor = reg_factor
            
    def init_MFA(self):
        for k in range(self.K):
            self.Psi_[k] = np.diag(np.diag(self.covariances_[k]))
            D,V = np.linalg.eig(self.covariances_[k] - self.Psi_[k])
            indexes = np.argsort(D)[::-1]
            V = dot(V[:,indexes], np.diag(np.lib.scimath.sqrt(D[indexes])))
            self.Lambda_[k] = V[:,:self.n_fac]
            
            B_k = dot(self.Lambda_[k].T, inv(dot(self.Lambda_[k],self.Lambda_[k].T)+self.Psi_[k]))
            self.Lambda_[k] = dot(self.covariances_[k],dot(B_k.T, inv(np.eye(self.n_fac)- dot(B_k,self.Lambda_[k]) + dot(B_k,dot(self.covariances_[k],B_k.T)))))
            self.Psi_[k] = np.diag(np.diag(self.covariances_[k] - dot(self.Lambda_[k], dot(B_k,self.covariances_[k])))) + np.eye(self.D)*self.reg_factor
           

    def fit(self,x, max_iter = 10, init_type = 'kmeans', threshold = 1e-4, n_init = 5):
        self.x = x
        self.N = len(self.x) #number of datapoints
        self.threshold = threshold
        
        self.Ls = np.zeros((self.N,self.K)) #posterior probability of z
        self.zs = np.zeros((self.N,self.K)) #posterior probability of z

        
        best_params = ()
        Lmax = -np.inf

        for it in range(n_init):
            if init_type == 'kmeans':
                self.init_kmeans()
            elif init_type == 'random':
                self.init_random()

            self.init_MFA()
                
            for i in range(max_iter):
                print('Iteration {}'.format(i))
                
                tic = time.time()
                self.expectation()
                toc = time.time()
                #print 'Expectation computation', toc-tic
            
                tic = time.time()
                self.maximization_1()
                toc = time.time()
                #print 'Maximization 1 computation', toc-tic

                #self.expectation()
                
                tic = time.time()
                self.maximization_2()
                toc = time.time()
                #print 'Maximization 2 computation', toc-tic

                print(self.L)
                if np.abs(self.prev_L-self.L) < self.threshold:
                    break
                    
            if self.L > Lmax:
                Lmax = self.L
                best_params = (self.L, self.weights_.copy(), self.means_.copy(), self.covariances_.copy(), self.zs.copy(), self.Ns.copy())
            
        #return the best result
        self.L = Lmax
        self.weights_ = best_params[1]
        self.means_ = best_params[2]
        self.covariances_ = best_params[3]
        self.zs = best_params[4]
        self.Ns = best_params[5]
        print('Obtain best result with Log Likelihood: ' + str(self.L))

    def maximization_1(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 
            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]  

    def maximization_2(self):
        #update covariance
        for k in range(self.K):
            x_reduce_mean = self.x-self.means_[k,:]
            #S_k = dot(x_reduce_mean.T, dot(np.diag(self.zs[:,k]), x_reduce_mean))
            S_k = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            B_k = dot(self.Lambda_[k].T, inv(dot(self.Lambda_[k],self.Lambda_[k].T)+self.Psi_[k]))
            
            
            self.Lambda_[k] = dot(S_k,dot(B_k.T, inv(np.eye(self.n_fac)- dot(B_k,self.Lambda_[k]) + dot(B_k,dot(S_k,B_k.T)))))
            self.Psi_[k] = np.diag(np.diag(S_k - dot(self.Lambda_[k], dot(B_k,S_k)))) + np.eye(self.D)*self.reg_factor
            self.covariances_[k] = dot(self.Lambda_[k], self.Lambda_[k].T) + self.Psi_[k]
        

class MPPCA(GMM):
    def __init__(self, D = 1, K = 2, n_fac = 1, reg_factor = 1e-6):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        
        self.Lambda_ = np.array([np.zeros((D,n_fac)) for i in range(K)])
        self.Psi_ = np.array([np.eye(D) for i in range(K)])
        self.sigma_ = np.ones(self.K)*1e-4
        
        self.n_fac = n_fac
        self.reg_factor = reg_factor
        

            
    def init_MPPCA(self):
        for k in range(self.K):
            self.sigma_[k] = np.trace(self.covariances_[k])/self.D
            print(self.sigma_[k])
            D,V = np.linalg.eig(self.covariances_[k] - np.eye(self.D)*self.sigma_[k])
            indexes = np.argsort(D)[::-1]
            print(D)
            V = dot(V[:,indexes], np.diag(np.lib.scimath.sqrt(D[indexes])))
            self.Lambda_[k] = V[:,:self.n_fac]
                       

    def fit(self,x, max_iter = 10, init_type = 'kmeans', threshold = 1e-4, n_init = 5):
        self.x = x
        self.N = len(self.x) #number of datapoints
        self.Ls = np.zeros((self.N,self.K)) #posterior probability of z
        self.zs = np.zeros((self.N,self.K)) #posterior probability of z

        self.threshold = threshold
        
        best_params = ()
        Lmax = -np.inf

        for it in range(n_init):
            if init_type == 'kmeans':
                self.init_kmeans()
                print(self.means_)
                print(self.covariances_)
                print(self.sigma_)
            elif init_type == 'random':
                self.init_random()

            self.init_MPPCA()
            print(self.sigma_)
                
            for i in range(max_iter):
                print('Iteration {}'.format(i))                
                #tic = time.time()
                self.expectation()
                #toc = time.time()
                #print 'Expectation computation', toc-tic
            
                #tic = time.time()
                self.maximization_1()
                #toc = time.time()
                #print 'Maximization 1 computation', toc-tic

                #self.expectation()
                
                #tic = time.time()
                self.maximization_2()
                #toc = time.time()
                #print 'Maximization 2 computation', toc-tic

                print(self.L)
                if np.abs(self.prev_L-self.L) < self.threshold:
                    break
                    
            if self.L > Lmax:
                Lmax = self.L
                best_params = (self.L, self.weights_.copy(), self.means_.copy(), self.covariances_.copy(), self.zs.copy(), self.Ns.copy())
            
        #return the best result
        self.L = Lmax
        self.weights_ = best_params[1]
        self.means_ = best_params[2]
        self.covariances_ = best_params[3]
        self.zs = best_params[4]
        self.Ns = best_params[5]
        print('Obtain best result with Log Likelihood: {}'.format(self.L))

    def maximization_1(self):
        for k in range(self.K):
            #update weight
            self.weights_[k] = self.Ns[k]/self.N 
            #update mean
            self.means_[k,:] = np.dot(self.zs[:,k].T, self.x)/self.Ns[k]  

    def maximization_2(self):
        #update covariance
        self.S = []
        self.M = []

        for k in range(self.K):
            x_reduce_mean = self.x-self.means_[k,:]
            #S_k = dot(x_reduce_mean.T, dot(np.diag(self.zs[:,k]), x_reduce_mean))
            S_k = dot(np.multiply(x_reduce_mean.T, self.zs[:,k][None,:]), x_reduce_mean)/self.Ns[k] + np.eye(self.D)*self.reg_factor
            M_k = dot(self.Lambda_[k].T,self.Lambda_[k]) + np.eye(self.n_fac)*self.sigma_[k]
            
            Lambda_k = dot(S_k,dot(self.Lambda_[k], inv(np.eye(self.n_fac)*self.sigma_[k] + \
                                dot(inv(M_k),dot(self.Lambda_[k].T,dot(S_k, self.Lambda_[k]) )  ))))
            self.S.append(S_k)
            self.M.append(M_k)
            
            
            self.sigma_[k] = np.trace(S_k-dot(S_k,dot(self.Lambda_[k],dot(inv(M_k),Lambda_k.T))))/self.D
            self.Psi_[k] = np.eye(self.D)*self.sigma_[k]
            
            self.Lambda_[k] = Lambda_k.copy()
            
            self.covariances_[k] = dot(self.Lambda_[k], self.Lambda_[k].T) + self.Psi_[k]
                   
        
class GMR():
    def __init__(self, GMM, n_in, n_out):
        self.GMM = GMM
        self.n_in = n_in
        self.n_out = n_out
        #segment the gaussian components
        self.mu_x = []
        self.mu_y = []
        self.sigma_xx = []
        self.sigma_yy = []
        self.sigma_xy = []
        self.sigma_xyx = []
        for k in range(self.GMM.n_components):
            self.mu_x.append(self.GMM.means_[k][0:self.n_in])        
            self.mu_y.append(self.GMM.means_[k][self.n_in:])        
            self.sigma_xx.append(self.GMM.covariances_[k][0:self.n_in, 0:self.n_in])        
            self.sigma_yy.append(self.GMM.covariances_[k][self.n_in:, self.n_in:])        
            self.sigma_xy.append(self.GMM.covariances_[k][0:self.n_in, self.n_in:])
            self.sigma_xyx.append(np.dot(self.sigma_xy[k].T,np.linalg.inv(self.sigma_xx[k])))
            
        self.mu_x = np.array(self.mu_x)
        self.mu_y = np.array(self.mu_y)
        self.sigma_xx = np.array(self.sigma_xx)
        self.sigma_yy = np.array(self.sigma_yy)
        self.sigma_xy = np.array(self.sigma_xy)
        self.sigma =[self.sigma_yy[k]- np.dot(self.sigma_xy[k].T, \
            np.dot(np.linalg.inv(self.sigma_xx[k]), self.sigma_xy[k])) for k in range(self.GMM.n_components)]
        
    def predict(self,x):
        h = []
        mu = []        

        for k in range(self.GMM.n_components):
            h.append(self.GMM.weights_[k]*mvn(mean = self.mu_x[k], cov = self.sigma_xx[k]).pdf(x))
            mu.append(self.mu_y[k] + np.dot(self.sigma_xyx[k], x - self.mu_x[k]))
        
        h = np.array(h)
        h = h/np.sum(h)
        mu = np.array(mu)
        sigma = self.sigma
        
        sigma_one = np.zeros([self.n_out, self.n_out])
        mu_one = np.zeros(self.n_out)
        for k in range(self.GMM.n_components):
            sigma_one += h[k]*(sigma[k] + np.outer(mu[k],mu[k]))
            mu_one += h[k]*mu[k]
            
        sigma_one -= np.outer(mu_one, mu_one)
        return mu_one, sigma_one
    
realmin = np.finfo(np.double).tiny
realmax = np.finfo(np.double).max

class HMM(GMM):
    def __init__(self, D = 1, K = 2,  reg_factor = 1e-6):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        self.reg_factor =  reg_factor 

    def init_kmeans(self):
        x = self.xs.reshape(-1,self.xs.shape[-1])
        kMM = KMeans(n_clusters=self.K).fit(x)
        self.means_ = kMM.cluster_centers_
        for i in range(self.K):
            self.covariances_[i] = np.cov(x[kMM.labels_==i].T) + np.eye(self.D)*self.reg_factor
        
    def init_random(self):
        self.means_ = self.x[np.random.choice(len(self.x),size = self.K)]
        for i in range(self.K):
            self.covariances_[i] = np.cov(self.x.T)
            
    def init_kbins(self):
        x = self.xs.reshape(-1,self.xs.shape[-1])
        tsep = np.linspace(0,self.N, self.K+1).astype(int)
        idx = []
        for k in range(self.K):
            idx.append(np.concatenate([m*self.N + np.arange(tsep[k],tsep[k+1]) for m in range(self.M)]))
            self.means_[k] = np.mean(x[idx[k]],axis=0)
            self.covariances_[k] = np.cov(x[idx[k]].T) + np.eye(self.D)*self.reg_factor
        
    def init_HMM(self):
        self.weights_ = np.ones(self.K)/self.K
        self.Trans_ =  np.ones((self.K,self.K))/self.K
            
    def fit(self,xs, max_iter = 10, init_type = 'kmeans', threshold = 1e-6, n_init = 5):
        self.xs = xs
        self.M = len(self.xs) #number of samples
        self.N = len(self.xs[0]) #number of datapoints

        self.threshold = threshold
        
        best_params = ()
        Lmax = -np.inf
        for it in range(n_init):
            if init_type == 'kmeans':
                self.init_kmeans()
            elif init_type == 'random':
                self.init_random()
            elif init_type == 'kbins':
                self.init_kbins()
                

            self.init_HMM()
            
            for i in range(max_iter):
                print('Iteration {}'.format(i))
                self.expectation()
                self.maximization()
                print(self.L)
                if np.abs(self.prev_L-self.L) < self.threshold:
                    break
                    
            if self.L > Lmax:
                Lmax = self.L
                best_params = (self.L, self.weights_.copy(), self.means_.copy(), self.covariances_.copy())
            
        #return the best result
        self.L = Lmax
        self.weights_ = best_params[1]
        self.means_ = best_params[2]
        self.covariances_ = best_params[3]
        print('Obtain best result with Log Likelihood: {}'.format(self.L))
        

    def compute_messages(self,x,obs_slice=None):
        N = x.shape[0]
        alpha = np.zeros((N,self.K)) 
        beta = np.zeros((N,self.K))
        gamma = np.zeros((N,self.K))
        zeta = np.zeros((N,self.K, self.K))
        B = np.zeros((N,self.K))
        c = np.zeros(N)
        
        #compute emission probabilities
        if obs_slice is not None:
            means_slice, covariances_slice = self.get_marginal(obs_slice)
            for k in range(self.K):
                B[:,k] = mvn(mean=means_slice[k], cov=covariances_slice[k]).pdf(x)            
        else:
            for k in range(self.K):
                B[:,k] = mvn(mean=self.means_[k], cov=self.covariances_[k]).pdf(x)

        #compute alpha
        alpha[0,:] = self.weights_ * B[0,:]
        c[0] = 1./(np.sum(alpha[0])+realmin)
        alpha[0] *= c[0]
        for n in range(1,N):
            alpha[n,:] = B[n,:]*np.dot(alpha[n-1,:], self.Trans_)
            c[n] = 1./(np.sum(alpha[n])+realmin)
            alpha[n] *= c[n]

        #compute beta
        beta[-1,:] = np.ones(self.K)*c[-1]
        for n in range(N-1,0,-1):
            beta[n-1,:] = np.dot(beta[n,:]*B[n,:], self.Trans_.T)
            beta[n-1,:] = (beta[n-1,:]*c[n-1])
                
                
        #compute likelihood
        L = -np.sum(np.log(c))

        #compute gamma
        for n in range(N):
            gamma[n,:] = alpha[n,:]*beta[n,:]/np.sum(alpha[n,:]*beta[n,:])

        #compute zeta
        for n in range(N-1):
            for k in range(self.K):
                zeta[n][k,:] = alpha[n,k]*B[n+1,:]*beta[n+1,:]*self.Trans_[k,:]

        return B,alpha,beta,gamma,zeta,L,c
            
    def expectation(self):
        self.prev_L = self.L

        self.alphas = np.zeros((self.M, self.N,self.K)) 
        self.betas = np.zeros((self.M, self.N,self.K))
        self.gammas = np.zeros((self.M, self.N,self.K))
        self.zetas = np.zeros((self.M, self.N,self.K, self.K))
        self.Bs = np.zeros((self.M,self.N,self.K))
        self.L = np.zeros(self.M)
        self.cs = np.zeros((self.M, self.N))
        
        for m in range(self.M):
            self.Bs[m], self.alphas[m], self.betas[m], self.gammas[m], self.zetas[m], self.L[m], self.cs[m] = \
                                                self.compute_messages(self.xs[m])
    
        self.L = np.mean(self.L)
             
    def maximization(self):
        x_mat = self.xs.reshape(-1,self.xs.shape[-1])
        gamma_mat = self.gammas.reshape(-1,self.gammas.shape[-1])
        zeta_mat = self.zetas.reshape(-1,self.zetas.shape[-2], self.zetas.shape[-1])
        self.x_mat = x_mat
        self.gamma_mat = gamma_mat
        self.zeta_mat = zeta_mat
        
        for k in range(self.K):
            #update weights
            self.weights_[k] =  np.mean(self.gammas[:,0,k]/np.sum(self.gammas[:,0,:],axis=1)) 

            means_k = np.zeros(self.D)
            sigma_k = np.zeros((self.D,self.D))
            
            #self.means_[k,:] 
            #self.covariances_[k,:]         
            
            for m in range(self.M):
                #update mean
                means_k += np.dot(self.gammas[m][:,k].T, self.xs[m])

                #update covariance
                x_reduce_mean = self.xs[m]-self.means_[k,:]
                sigma_k += dot(np.multiply(x_reduce_mean.T, self.gammas[m][:,k][None,:]), x_reduce_mean) 
            
            self.means_[k,:] = means_k / np.sum(self.gammas[:,:,k])
            self.covariances_[k,:] = sigma_k/np.sum(self.gammas[:,:,k])        + np.eye(self.D)*self.reg_factor
        
        self.Trans_ = np.sum(zeta_mat,axis=0)
        for k in range(self.K):
            self.Trans_[k,:] /= np.sum(self.Trans_[k,:])
        #/np.sum(zeta_mat,axis=(0,2))[:,None]
        
    def viterbi(self,x):
        self.delta = np.zeros((x.shape[0],self.K))
        self.psi = np.zeros((x.shape[0]-1,self.K))
        self.B = np.zeros((x.shape[0],self.K))
        
        self.opt_seq = np.zeros(x.shape[0]).astype(int)
        #compute emmision probabilities
        for k in range(self.K):
            self.B[:,k] = mvn(mean=self.means_[k], cov=self.covariances_[k]).logpdf(x)

        #initialise at n=0
        self.delta[0,:] = np.log(self.weights_) + self.B[0,:]
        for n in range(1,x.shape[0]):
            temp_vals = np.log(self.Trans_) + self.delta[n-1,:][:,None]
            self.delta[n,:] = self.B[n,:] + np.max(temp_vals, axis=0)
            self.psi[n-1,:] = np.argmax(temp_vals,axis=0)
        
        #find the optimal sequence
        self.opt_seq[-1] = np.argmax(self.delta[-1,:])
        for n in range(x.shape[0]-1, 0, -1):
            self.opt_seq[n-1] = self.psi[n-1,self.opt_seq[n]]
        return self.opt_seq
    
    def condition(self,x_in,dim_in,dim_out,h=None, return_gmm = False):
        _,alpha,_,_,_,_,_ =  self.compute_messages(x_in, dim_in)
        print(alpha.shape)
        x_out = []
        for i,x in enumerate(x_in):
            pred,cov = GMM.condition(self,x[None,:],dim_in,dim_out,h=alpha[i],return_gmm=return_gmm)
            x_out += [pred]
        return np.asarray(x_out)
    
    def plot(self):
        fig,ax = plt.subplots()
        plot_GMM(self.means_, self.covariances_, ax)
        
        
class HSMM(HMM):
    def compute_durations(self, xs):
        self.state_durs = dict()
        for k in range(self.K):
            self.state_durs[k] = []

        #compute the optimal state sequences for the data xs
        opt_seqs = []
        for x in xs:
            opt_seqs += [self.viterbi(x)]
        
        #compute the state duration for each sequence    
        for opt_seq in opt_seqs:
            cur_state = opt_seq[0]
            cur_count = 1
            for s in opt_seq[1:]:
                if cur_state == s:
                    cur_count += 1
                else:
                    self.state_durs[cur_state] += [cur_count]
                    cur_state = s
                    cur_count = 1
            #for the last state
            self.state_durs[cur_state] += [cur_count]
        
        #compute the duration probability as Gaussian Distribution for each state
        self.means_pd_ = np.zeros(self.K)
        self.sigmas_pd_ = np.zeros(self.K)
        for k in range(self.K):
            self.means_pd_[k] = np.mean(self.state_durs[k])
            self.sigmas_pd_[k] = np.var(self.state_durs[k])
            
    def compute_forward_messages_HSMM(self,x, obs_slice = None):
        if obs_slice is not None:
            means_, covariances_ = self.get_marginal(obs_slice)
        else:
            means_, covariances_ = (self.means_, self.covariances_)
        
        
        #precompute duration probabilities at each time step
        num_dur_max = int(2.*self.N/self.K) #the number of maximum duration that will be considered
        self.Pd = np.zeros((num_dur_max,self.K))
        for k in range(self.K):
            self.Pd[:,k] = mvn(mean=self.means_pd_[k], cov=self.sigmas_pd_[k]).pdf(np.arange(num_dur_max))
            self.Pd[:,k] /= np.sum(self.Pd[:,k])
            
        #compute the observation likelihood
        B = np.zeros((self.N,self.K))
        for k in range(self.K):
            B[:,k] = mvn(mean = means_[k], cov=covariances_[k]).pdf(x)
        
            
        h = np.zeros((self.N,self.K))
        c = np.zeros(self.N)
        c[0] = 1.
        
        for t in range(self.N):
            for k in range(self.K):
                if t < num_dur_max:
                    #o_tmp = np.prod(c[:t+1]*mvn(mean = means_[k], cov=covariances_[k]).pdf(x[:t+1]))
                    o_tmp = np.prod(c[:t+1]*B[:t+1,k])
                    h[t,k] = self.weights_[k]*self.Pd[t,k]*o_tmp
                
                for d in range(np.min([t,num_dur_max])):
                    #o_tmp = np.prod(c[t-d:t+1]*mvn(mean = means_[k], cov=covariances_[k]).pdf(x[t-d:t+1]))
                    o_tmp = np.prod(c[t-d:t+1]*B[t-d:t+1,k])
                    h[t,k] += np.dot(h[t-d-1,:],self.Trans_[:,k])*self.Pd[d,k]*o_tmp
            if t < self.N-1:
                c[t+1] = 1/np.sum(h[t,:])
            
        h = h/np.sum(h,axis=1)[:,None]
        self.num_dur_max = num_dur_max
        return h,c
    def condition(self,x_in,dim_in,dim_out,h=None, return_gmm = False):
        h,_ =  self.compute_forward_messages_HSMM(x_in, dim_in)
        x_out = []
        for i,x in enumerate(x_in):
            pred,cov = GMM.condition(self,x[None,:],dim_in,dim_out,h=h[i],return_gmm=return_gmm)
            x_out += [pred]
        return np.asarray(x_out)
    def edit_trans(self):
        #Remove self transition and normalize
        self.Trans_ -= np.diag(np.diag(self.Trans_))

        #normalize
        for k in range(self.K):
            self.Trans_[k,:] /= np.sum(self.Trans_[k,:])
