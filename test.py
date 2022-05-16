from stepwise_lr import SLR
import numpy as np, pandas as pd
from scores import mean_squared_error

def getTestData(n_features=100, n_informative=25, n_samples=10000, random_state=0, sigmaStd=.0):
    #generate random dataset
    from sklearn.datasets import make_regression
    np.random.seed(random_state)
    X,y = make_regression(n_samples=n_samples, n_features=n_features,
                             n_informative=n_informative, shuffle=False, random_state=random_state)
    cols = ['I_'+str(i) for i in range(n_informative)]
    cols += ['N_'+str(i) for i in range(n_features-n_informative)]
    X,y = pd.DataFrame(X,columns=cols),pd.Series(y)
    return X,y




params_dict = {
    'error_f':mean_squared_error
}

if __name__=="__main__":
    X,y = getTestData(40,5,30,10000,sigmaStd=.1)
    _ = SLR(X, y, k_folds=5, greater_is_better=True, **params_dict)._run()