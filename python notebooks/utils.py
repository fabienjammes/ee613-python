from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
import numpy as np

from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn
class GMM():
    def __init__(self, D = 1, K = 2):
        self.D = D #number of dimensions
        self.K = K #number of mixture components
        self.L = -np.inf #total log likelihood
        self.weights_ = np.ones(K)/K
        self.means_ = np.random.rand(K,D)
        self.covariances_ = np.array([np.eye(D) for i in range(K)])
        
    def init_kmeans(self):
        kMM = KMeans(n_clusters=self.K).fit(self.x)
        self.means_ = kMM.cluster_centers_
        for i in range(self.K):
            self.covariances_[i] = np.cov(self.x[kMM.labels_==i].T)
        
    def init_random(self):
        self.means_ = self.x[np.random.choice(len(self.x),size = self.K)]
        for i in range(self.K):
            self.covariances_[i] = np.cov(self.x.T)

    def fit(self,x, max_iter = 10, init_type = 'kmeans', threshold = 1e-4, n_init = 5):
        self.x = x
        self.N = len(self.x) #number of datapoints
        self.threshold = threshold
        
        best_params = ()
        Lmax = -np.inf
        for it in range(n_init):
            if init_type == 'kmeans':
                self.init_kmeans()
            elif init_type == 'random':
                self.init_random()

            for i in range(max_iter):
                print 'Iteration ' + str(i)
                self.expectation()
                self.maximization()
                print self.L
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
        print 'Obtain best result with Log Likelihood: ' + str(self.L)
        
    def expectation(self):
        self.Ls = np.zeros((self.N,self.K)) #posterior probability of z
        self.zs = np.zeros((self.N,self.K)) #posterior probability of z
        
        for i in range(self.N):
            for k in range(self.K):
                self.Ls[i,k] = self.weights_[k]*mvn.pdf(self.x[i,:],mean = self.means_[k], cov=self.covariances_[k])

            self.zs[i,:] = self.Ls[i,:]/np.sum(self.Ls[i,:]) #normalize
        
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
            sigma_k = np.zeros((self.D,self.D))
            for i in range(self.N):
                sigma_k += self.zs[i,k]*np.outer(self.x[i,:]-self.means_[k,:], self.x[i,:].T-self.means_[k,:].T)
            sigma_k /= self.Ns[k]
            self.covariances_[k,:] = sigma_k        
        
    def plot(self):
        fig,ax = plt.subplots()
        plot_GMM(self.means_, self.covariances_, ax)
        
from scipy.stats import multivariate_normal as mvn
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

def plot_gaussian_1D(mu, sigma, ax,offset = None, bound= None, color = [.4,.4,.4], alpha = 1., normalize = True, prior = None, label = 'label', orient = 'h'):
    n = 100
    if bound is None:
        bound = [mu-2*sigma, mu+2*sigma]
        
    x = np.linspace(bound[0], bound[1], n)
    y = normal_dist(loc=mu, scale=sigma).pdf(x)
    if normalize:
        y = y/np.max(y)
    if prior is not None:
        y *= prior

    if offset is not None:
        y += offset
    
    if orient == 'h':
        poly_data = np.vstack([x,y]).T
        axis_limit = [bound[0], bound[1], 0, np.max(y)]
    else:
        poly_data = np.vstack([y,x]).T
        axis_limit = [0, np.max(y), bound[0], bound[1]]
        
    polygon = Polygon(poly_data,False,color=color,label=label, alpha=alpha)
    #plt.plot(y,x)
    ax.add_patch(polygon)

    plt.axis(axis_limit)
    return x,y

def plot_dist_1D(x,y,ax, color = [.4,.4,.4], alpha = 1.,label = 'label'):
    poly_data = np.vstack([x,y]).T
    axis_limit = [bound[0], bound[1], 0, np.max(y)]
    polygon = Polygon(poly_data,False,color=color,label=label, alpha=alpha)
    ax.add_patch(polygon)
    plt.axis(axis_limit)
    return

def plot_gaussian_2D(mu, sigma,ax,color=[0.7,0.7,0.7],alpha=1.0, label='label'):
    eig_val, eig_vec = np.linalg.eigh(sigma)
    std = np.sqrt(eig_val)*3
    angle = np.arctan2(eig_vec[1,0],eig_vec[0,0])
    ell = Ellipse(xy = (mu[0], mu[1]), width=std[0], height = std[1], angle = np.rad2deg(angle))
    ell.set_facecolor(color)
    ell.set_alpha(alpha)
    ell.set_label(label)
    ax.add_patch(ell)
    return


def plot_data_2D(x, y_true, y_pred, colors = ['r', 'k', 'b'], title = 'Linear regression', alphas = None):
    if alphas is None:
        alphas = np.ones(len(x))
    #plot the predicted data
    plt.plot(x,y_pred, '-' + colors[0])
    
    #plot the error bar and the true data
    for i in range(len(x)):
        #plot the true data
        plt.plot(x[i],y_true[i], '.' + colors[1])#,alpha = alphas[i])
        plt.plot([x[i],x[i]], [y_true[i], y_pred[i]], '-'+colors[2], alpha = alphas[i])
    plt.xlabel(u'x\u2081')
    plt.ylabel(u'y\u2081')
    
    plt.title(title)
    
    return

def plot_GMM(mus, sigmas, ax, colors = None, alphas = None, labels = None):
    n = len(mus)
    for i in range(n):
        plot_gaussian_2D(mus[i], sigmas[i], ax)
    return

import matplotlib.pyplot as plt 

def plot_with_covs_1D(x, y, cov, ax):
    y_low = y - 2*np.sqrt(cov)
    y_up = y + 2*np.sqrt(cov)
    y_up = y_up[::-1]
    
    x_1 = np.concatenate([x, x[::-1]])
    y_1 = np.concatenate([y_low, y_up])
    xy = np.vstack([x_1,y_1]).T
    poly = Polygon(xy,alpha=0.4)
    ax.add_patch(poly)
    
    plt.plot(x,y,'-r')