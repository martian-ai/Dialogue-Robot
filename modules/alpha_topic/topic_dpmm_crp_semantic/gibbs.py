# Copyright (c) 2017, Danyang 'Frank' Li <danyangl@mtu.edu>

from __future__ import division
from distribution import UnivariateGaussian
import numpy as np

class dpmm_gibbs_base(object):
    def __init__(self, init_K=5, x=[], alpha_prior=None):
        """
        init_K 主题个数 int
        x 数据 []
        alpha_prior 
        """
        #Convert python array to numpy array
        self.x = np.asarray(x)
        self.K = init_K

        self.nn = np.ones(self.K)
        self.alpha_prior = alpha_prior
        self.alpha_0 = 1
        #np.random.gamma(self.alpha_prior['a'],self.alpha_prior['b'])
        # init zz randomly
        self.zz = np.random.randint(init_K, size=len(self.x)) # 为每个数据 随机分配一个主题
        print(self.zz)
        self.mu_0 = 1
        self.mu = np.ones(self.K)
        self.components = [mixture_component(ss=[], distn=UnivariateGaussian(mu=mu_i)) for mu_i in self.mu] # 构建默认K 个 component 
        for idx, c in enumerate(self.components):
            c.ss = self.x[self.zz == idx]
            self.nn[idx] = len(c.ss)
        self.n = len(self.x)


class collapsed_dpmm_gibbs(dpmm_gibbs_base):
    def __init__(self, init_K=5, x=[], alpha_prior = None, observation_prior=None,):
        super(collapsed_dpmm_gibbs, self).__init__(init_K, x, alpha_prior) # 对 init_k, x, alpha_prior 赋值
        self.observation_prior = observation_prior
        #add a new empty component
        new_mu = np.random.normal(self.observation_prior['mu'], self.observation_prior['sigma'], 1);  # 产生新的 mu
        new_component = mixture_component(ss=[], distn=UnivariateGaussian(mu=new_mu)) # 新构建一个 component 
        self.components = np.append(self.components, new_component)
        #print (UnivariateGaussian.epsilon_log_univariate_normal(self,-12,2) - UnivariateGaussian.epsilon_log_univariate_normal(self,1,1))

    def sample_z(self):
        """
        通过gibbs 采样确定 数据的主题分布
        """
        print('输入的数据', self.x)
        print('对应的主题', self.zz)
        for idx, x_i in enumerate(self.x): # 输入数据
            print('当前数据', x_i)
            print('当前下标', idx)
            kk = self.zz[idx] # 输入数据的主题
            # Clean mixture components
            print('当前数据的主题', kk)
            temp_zz, = np.where(self.zz == kk)
            print('与当前数据主题相同的数据的下标', temp_zz)
            # print('----')
            # print len(self.components[kk].ss)
            temp_zz = np.setdiff1d(temp_zz, np.array([idx])) # 在ar1中但不在ar2中的已排序的唯一值
            print('当前主题其他元素的下标', temp_zz)
            self.nn[kk] -= 1 # 当前主题计数减少1
            temp_ss = self.x[temp_zz] # 当前主题对应的元素，减去当前元素
            print('当前主题对应的元素，减去当前元素', temp_ss)
            #print len(temp_ss)
            self.components[kk].ss = temp_ss
            if (len(temp_ss) == 0):
                print('component deleted')
                print('length of components', len(self.components))
                self.components = np.delete(self.components, kk)
                print('length of components', len(self.components))

                self.K = len(self.components)
                self.nn = np.delete(self.nn,kk)
                zz_to_minus_1 = np.where(self.zz > kk)
                self.zz[zz_to_minus_1] -= 1
            pp = np.log(np.append(self.nn, self.alpha_0))
            for k in range(0, self.K):
                pp[k] = pp[k] + self.log_predictive(self.components[k],x_i)
                #print(self.log_predictive(self.components[k],x_i))
            pp = np.exp(pp - np.max(pp))
            pp = pp/np.sum(pp)
            sample_z = np.random.multinomial(1, pp, size=1)
            # print(x_i)
            z_index = np.where(sample_z == 1)[1][0]
            self.zz[idx] = z_index
            if(z_index == len(self.components) - 1):
                print('component added')
                new_mu = np.random.normal(0.5 * x_i, 0.5, 1)
                new_component = mixture_component(ss=[x_i], distn=UnivariateGaussian(mu=new_mu))
                self.components = np.append(self.components, new_component)
                self.K = len(self.components)
                self.nn = np.append(self.nn, 1)
            else:
                self.components[z_index].ss = np.append(self.components[z_index].ss, x_i)
                self.nn[z_index] += 1

        print('----Summary----')
        print(self.zz)
        # print self.nn
        # for component in self.components:
        #     component.print_self()

    def sample_alpha_0(self):
        """
        更新 gamma 分布 的 参数，并重新采样
        """
        #Escobar and West 1995
        eta = np.random.beta(self.alpha_0 + 1,self.n,1)
        #Teh HDP 2005
        #construct the mixture model
        pi = self.n/self.alpha_0
        pi = pi/(1+pi)
        s = np.random.binomial(1,pi,1)
        #sample from a two gamma mixture models
        self.alpha_0 = np.random.gamma(self.alpha_prior['a'] + self.K - s, 1/(self.alpha_prior['b'] - np.log(eta)), 1)
        # print(self.alpha_0)

    def log_predictive(self,component, x_i):
        ll = UnivariateGaussian.epsilon_log_univariate_normal(self, self.observation_prior['mu'] + np.sum(component.get_ss()) + x_i , self.observation_prior['sigma'] + component.get_n_k_minus_i() + 1) - \
             UnivariateGaussian.epsilon_log_univariate_normal(self, self.observation_prior['mu'] + np.sum(component.get_ss()), self.observation_prior['sigma'] + component.get_n_k_minus_i())
        return ll


class mixture_component(object):
    def __init__(self, ss, distn):
        self.ss = ss
        self.distn = distn
        if(len(ss)> 0):
            self.n_k_minus_i = len(ss) - 1
        else:
            self.n_k_minus_i = 0

    def get_n_k_minus_i(self):
        """
        返回当前簇含有样本的个数
        """
        if (len(self.ss) > 1):
            self.n_k_minus_i = len(self.ss) - 1
        else:
            self.n_k_minus_i = 0
        return self.n_k_minus_i

    def get_ss(self):
        return self.ss

    def print_self(self):
        #print(self.ss)
        print('Mu: '+ str(self.distn.mu))