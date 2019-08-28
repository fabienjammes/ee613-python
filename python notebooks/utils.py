from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
import numpy as np

from sklearn.cluster import KMeans
class GMM():
    def __init__(self, n_comp = 10, D = 1, K = 2, weight_concentration = 0.1):
        self.n_comp = n_comp
        self.D = D
        self.K = K

        #pi params
        self.alpha0 = np.ones(K)*weight_concentration
        self.alpha = np.copy(self.alpha0)

        #mu and sigma params
        self.betha0 = 0.0001
        self.betha = np.ones(self.K)
        self.mu0 = np.zeros(self.D)
        self.v0 = self.D
        self.W0 = np.eye(self.D)/(self.v0)
        #
        self.W = [np.eye(self.D) for i in range(self.K)]
        self.v = np.ones(self.K)*self.D
        
        self.reg_covar = 1e-06
        
    def fit(self,x, num_iter = 10,restart=True, stochastic=True, batch_size = 2000):
        self.x = x
        self.stochastic = stochastic
        self.batch_size = batch_size
        cov = np.cov(x.T)
        if restart:
            kMM = KMeans(n_clusters=self.K).fit(x)
            self.mu = kMM.cluster_centers_
            #self.mu = x[np.random.choice(len(x),size = self.K)]
        for i in range(num_iter):
            print 'Iteration ' + str(i)
            self.expectation()
            #print self.ro, self.xks, self.Nks
            self.maximization()
            #print self.betha, self.v, self.W
            
        self.sigma = []
        for k in range(self.K):
            sigma = np.linalg.inv(self.v[k]*self.W[k])
            self.sigma.append(sigma)

        self.weights = self.alpha/np.sum(self.alpha)


    def expectation(self):
        if self.stochastic:
            indexes = np.arange(0, len(self.x),1)
            np.random.shuffle(indexes)
            indexes = indexes[0:self.batch_size]
            data = self.x[indexes]
            N = self.batch_size
        else:
            data = self.x
            N = data.shape[0]
            
        self.ln_ro = np.zeros([N,self.K])
        self.ro = np.zeros([N,self.K])
        
        print 'Calculating ro'
        tic = time.time()
        for k in range(self.K):
            E_s = self.D*np.log(2) + np.log(np.linalg.det(self.W[k])) + np.sum([digamma((self.v[k] + 1 - i)/2.) \
                                                                           for i in range(self.D)])
            E_pi = digamma(self.alpha[k]) - digamma(np.sum(self.alpha))
            E_2 = self.D*np.log(2*np.pi)
            for n in range(data.shape[0]):
                E_ms = self.D/self.betha[k] + self.v[k]*np.dot(data[n] - self.mu[k], np.dot(self.W[k], \
                                                                                data[n] - self.mu[k]))
                self.ln_ro[n,k] = E_pi + 0.5*E_s - 0.5*E_2 - 0.5*E_ms

        for n in range(N):        
            self.ln_ro[n,:] = log_normalize(self.ln_ro[n,:])
            self.ro[n,:] = np.exp(self.ln_ro[n,:])
        toc = time.time()
        print toc-tic


        self.Nks = np.array([1e-6 + np.sum(self.ro[:,k]) for k in range(self.K)])

        self.xks = np.dot(self.ro.T, data)
        for k in range(self.K):
            self.xks[k,:] /= self.Nks[k]


        print 'Calculating Sk' 
        self.Sks = []
        for k in range(self.K):
            Sk = np.zeros([self.D,self.D])
            for n in range(N):
                Sk += self.ro[n,k]*np.outer(data[n]-self.xks[k], data[n]-self.xks[k])
            Sk /= self.Nks[k]
            self.Sks.append(Sk)

        toc = time.time()
        print toc-tic

             
    def maximization(self):
        self.alpha = self.alpha0 + self.Nks
        self.betha = self.betha0 + self.Nks

        for k in range(self.K):
            self.mu[k] = (self.betha0*self.mu0 + self.Nks[k]*self.xks[k])/self.betha[k]
            Wk_inv = np.linalg.inv(self.W0 + self.reg_covar*np.eye(self.D)) + self.Nks[k]*self.Sks[k] + np.outer(self.xks[k]-self.mu0, \
                                    self.xks[k]-self.mu0)*self.betha0*self.Nks[k]/(self.betha0+self.Nks[k])
            self.W[k] = np.linalg.inv(Wk_inv+ self.reg_covar*np.eye(self.D))
            self.v[k] = self.v0 + self.Nks[k]
            
    def plot(self):
        fig,ax = plt.subplots()
        plot_gaussian(self.mu, self.sigma, ax, self.weights )

        
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