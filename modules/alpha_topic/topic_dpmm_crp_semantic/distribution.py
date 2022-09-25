# Copyright (c) 2017, Danyang 'Frank' Li <danyangl@mtu.edu>
from __future__ import division
import numpy as np

#fix var = 1 standard normal distribution
#with prior mu conjugate to a normal distribution
class UnivariateGaussian(object):
    """
    单变量 高斯
    """

    def __init__(self,mu=None):
        self.mu = mu

    def rvs(self, size=None):
        """
        normal 正太分布
        从 均值为 self.mu 方差为 1 的 正太分布中 采样 产生 size 大小的 矩阵
        """
        return np.random.normal(self.mu,1,size)

    def set_mu(self,mu=None):
        self.mu = mu

    def sample_new_mu(self,x):
        return np.random.normal(0.5*x, 0.5, 1)

    def log_likelihood(self,x):
        x = np.reshape(x, (-1, 1))
        return (-0.5 * (x - self.mu) ** 2 - np.log(2 * np.pi )).ravel() # ravel 将多维度数组 转化为 1 维度

    @staticmethod
    def epsilon_log_univariate_normal(self, mu, sigma):
        """
        mu
        sigma
        """
        return np.log(1/(sigma * np.sqrt(2*np.pi))) - mu**2/2*sigma**2



