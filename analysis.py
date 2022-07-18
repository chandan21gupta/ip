#from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
#import pandas as pd
#import numpy as np
#
#df = pd.DataFrame({
#    'T': [5, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
#    'E': [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0],
#    'var': [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
#    'age': [4, 3, 9, 8, 7, 4, 4, 3, 2, 5, 6, 7],
#})
#
#aft = WeibullAFTFitter()
#aft.fit(df, 'T', 'E')
#print(aft.params_)


from sklearn.linear_model import Ridge
import numpy as np
n_samples, n_features = 10, 5
rng = np.random.RandomState(0)
y = rng.randn(n_samples)
X = rng.randn(n_samples, n_features)
clf = Ridge(alpha=1.0)
clf.fit(X, y)
print(clf.coef_)
