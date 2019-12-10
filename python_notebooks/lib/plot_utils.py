from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse

from scipy.stats import norm as normal_dist

import numpy as np
from numpy import dot
from numpy.linalg import inv
from numpy.linalg import pinv
import time
import matplotlib.pyplot as plt 


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
    std = np.sqrt(eig_val)*2
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
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.title(title)
    
    return

def plot_GMM(mus, sigmas, ax, colors = None, alphas = None, labels = None):
    n = len(mus)
    if colors is None:
        colors = [[0.7,0.7,0.7]]*n
    if alphas is None:
        alphas = [1.]*n
    print alphas
        
    for i in range(n):
        if labels is None:
            plot_gaussian_2D(mus[i], sigmas[i], ax, color=colors[i],alpha=alphas[i])
        else:
            plot_gaussian_2D(mus[i], sigmas[i], ax,label=labels[i],color=colors[i], alpha = alphas[i])
    return


def plot_with_covs_1D(x, y, cov, ax):
    """
    Plot the graph of y against x, together with the covariance of y
    """
    
    y_low = y - 2*np.sqrt(cov)
    y_up = y + 2*np.sqrt(cov)
    y_up = y_up[::-1]
    
    x_1 = np.concatenate([x, x[::-1]])
    y_1 = np.concatenate([y_low, y_up])
    xy = np.vstack([x_1,y_1]).T
    poly = Polygon(xy,alpha=0.4)
    ax.add_patch(poly)
    
    plt.plot(x,y,'-r')
    