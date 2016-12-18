from mlcv.plotting import plotSVMparam as splot



splot('resultsSVM_linear_cost',mode='cost',name='linear')
splot('../../ResultsSVM_poly.pickle',mode='3d',name='poly')
splot('../../ResultsSVM_rbf_2nd.pickle',mode='2d',name='rbf')
splot('../../ResultsSVM_sigmoid_2nd_lighter.pickle',mode='3d',name='sigmoid')
splot('../../ResultsSVM_linear_cost.pickle',mode='cost',name='linear')
splot('../../ResultsSVM_poly_cost.pickle',mode='cost',name='poly')
splot('../../ResultsSVM_rbf_cost_2nd.pickle',mode='cost',name='rbf')
splot('../../ResultsSVM_sigmoid_cost_final.pickle',mode='cost',name='sigmoid')
