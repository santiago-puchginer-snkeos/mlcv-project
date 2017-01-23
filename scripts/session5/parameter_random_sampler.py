from sklearn.model_selection import ParameterSampler

param_grid = {'a': [1, 2], 'b': ['adam','holi','accuracy'], 'c':[1,10,100,1000]}
param_list = list(ParameterSampler(param_grid, n_iter=4))