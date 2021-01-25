# Expectation Maximization Algorithm
- Determine the log-likelihood function of complete data
	- observation data $O=(o_1,o_2,...o_T)$
	- hidden data $I=(i_1,i_2,...,i_T)$
	- complete data $(O,I)={o_1,o_2,...o_T,i_1,i_2,...,i_T}$
	- log-likelihood function of complete data log $P(O,I|\lambda)$

- E-Step

	- reference : cross entropy

		$$H(p,q) = - \sum_{x} p(x) log\ q(x)$$ 

	- Find Q function(Why define like this?) 
		$$Q(\lambda, \bar{\lambda}) = \sum_I logP(O,I|\lambda)P(O,I|\bar\lambda)$$

		- $\bar{\lambda}$ is the current estimate of the hmm parameter
		- $\lambda$ is the hmm parameter to maximize

	- Due to 
		$$ P( O, T | \lambda) = P( O | I, \lambda) P( I | \lambda) = 
		​    \pi_{i_1} a_{i_1i_2}a_{i_2i_3}...a_{i_{T-1} i_T}
		​    b_{i_1}(o_1)b_{i_2}(o_2)...b_{i_T}(o_T)
		$$

	- So
		$$Q(\lambda, \bar{\lambda}) = \sum_{I}log\pi_{i_1}P(O,I|\lambda)) + \sum_{I}\{\sum_{t=1}^{T-1}log a_{i_ti_{t+1}}\}P(O,I|\lambda)) + \sum_{I}\{\sum_{t=1}^{T-1}log b_{i_t}(o_t)\}P(O,I|\lambda))\ \ \ (1-1)$$

- M-Step

	- Maximize Q function to find model parameters $\pi, A, B$
	- Maximize the three terms above
		- First Item 
			- $\sum_{I}log\pi_{i_1}P(O,I|\lambda)) = \sum_{i=1}^{N}log\pi_iP(O,i_1=i|\bar\lambda )$
				- Constraints are $\sum_{i=1}^{N} \pi_i =1$
			- Lagrangian multiplier method
				$$ L =  \sum_{i=1}^{N}log\pi_iP(O,i_1=i|\bar\lambda ) + \gamma\{\sum_{i=1}^{N} \pi_i =1\}$$
				- Get Partial derivative
					$$ \frac{\partial L}{\partial \pi_i} = 0 \ \ \ (1-2)$$
				- Get 
					$$ P(O,i_1=i|\bar\lambda ) + \gamma\pi_i $$
				- Sum i then get
					$$ \gamma = - P(O|\bar\lambda) $$
				- Bring into Formula 1-2 then get
					$$ \pi_i = \frac{P(O,i_1=i| \bar\lambda )}{P(O|\bar\lambda))}\ \ \ (1-3)$$
		- Second Item
			- Constraints are $\sum_{j=1}^N a_{ij} =1$
			- Get 
				$$ a_{ij} = \frac{ \sum_{t=1}^{T-1} P(O,i_t=i,i_t+1 =j | \bar\lambda )}{P(O,i_t = i|\bar\lambda))}\ \ \ (1-4)$$
		- Third Item
			- Constraints are $\sum_{j=1}^{N} b_j(k) =1$
				$$ b_{j}(k) = \frac{ \sum_{t=1}^{T-1} P(O,i_t=j | \bar\lambda ) I(o_t = v_k)}{P(O,i_t = j|\bar\lambda))}\ \ \ (1-5)$$ 