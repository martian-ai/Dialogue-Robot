from gibbs import collapsed_dpmm_gibbs
from matplotlib import pyplot as plt
import numpy
plt.style.use('ggplot')

x = [ 4.0429277, 10.71686209, 10.73144389, 5.05700962, 4.70910861, \
      1.38603028, -12.87114683, 0.90842492, 2.26485196, 0.3287409, \
      1.85740593, -0.08981766,  0.11817958,  0.60973202,  1.88309994, \
      1.47112954,  0.77061995,  1.24543065,  1.92506892,  0.7578275, -30.12442321]

x1 = numpy.random.randint(0, 100, [21,10])
x2 = numpy.random.randint(100, 200, [21,10])
x3 = numpy.random.randint(200, 300, [21,10])

x = x1 + x2 + x3
print(len(x))

init_K = 3 # 主题个数 后续可以进一步改进
alpha_prior = {'a':1,'b':2} # gamma 分布的参数
observation_prior = {'mu':0,'sigma':10} # 正太分布的参数

collapsed_gibbs = collapsed_dpmm_gibbs(init_K,x,alpha_prior,observation_prior)
iter = 50
for i in range(0,iter):
    collapsed_gibbs.sample_z()
    collapsed_gibbs.sample_alpha_0()

print(len(collapsed_gibbs.components))
print(collapsed_gibbs.components[0].get_ss())
print(collapsed_gibbs.components[1].get_ss())
print(collapsed_gibbs.components[2].get_ss())
print(collapsed_gibbs.components[3].get_ss())
print(collapsed_gibbs.components[4].get_ss())
print(collapsed_gibbs.components[4].get_n_k_minus_i())
