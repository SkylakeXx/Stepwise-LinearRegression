from statsmodels.regression.linear_model import OLS
from sklearn_wrapper import SkWrapper
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from sklearn import metrics


class SLR:
    def __init__(self, x_df, y_df, k_folds=5, greater_is_better=True, *args, **kwargs):
        self.error_f = self.__default_err() if kwargs.get('error_f', None) is None else kwargs.get('error_f')
        self.nfolds = k_folds
        self.matrix = self.__init_matrix(x_df.columns)
        self.gib = greater_is_better
        self.iter_ls = [-np.inf if self.gib else np.inf]
        self.X = x_df
        self.y = y_df.values.flatten()
        self.iter = 0
        self.cols_ls = []
        self.is_done = False
        self.ols = SkWrapper(OLS)

    def __get_score(self, col_, X, y):
        # print(metrics.get_scorer_names())
        return np.mean(cross_val_score(self.ols, X, y, cv=5, scoring=self.error_f))

    def __get_cols(self):
        set_used_cols = set(self.cols_ls)
        set_free_cols = set(self.matrix.columns)-set_used_cols
        return list(set_free_cols)
    
    
    def _run(self):
        while self.iter<=len(self.matrix.columns):
            self.__run_iteration()
            if (self.is_done) or (self.iter>=self.X.shape[1]):
                break
        return self.__get_summary()


    def __run_iteration(self):
        iter_ = self.__get_iter()
        iter_columns_ = self.__get_cols()
        if len(self.cols_ls)>0:
            baseX = self.X[self.cols_ls].values
        else:
            baseX=np.array([], dtype=np.float32).reshape(self.y.shape[0],-1)
        top_score = -np.inf if self.gib else np.inf
        top_col = None
        for col_ in iter_columns_:
            by = self.X[[col_]].values
            X = np.hstack((baseX,by.reshape(-1,1)))
            y = self.y
            new_score = self.__get_score(col_, X, y)
            if self.__is_better(top_score, new_score):
                top_score = new_score
                top_col = col_
        if not self.__is_better(self.iter_ls[-1], top_score):
            self.is_done = True
            
        print(self.matrix.shape)
        print(top_col)
        print(iter_)
        print(top_score)
        print("\n")
        self.matrix.loc[iter_:, col_] = True
        self.iter_ls.append(top_score)
        self.cols_ls.append(top_col)    
    
    def __get_iter(self):
        self.iter = self.iter+1
        return self.iter-1
    
    def __is_better(self, baseline, new_record):
        if self.gib:
            return new_record>baseline
        else:
            return new_record<baseline 
        
    
    def __get_summary(self):
        return self.matrix, self.cols_ls, self.iter_ls 
    
    @staticmethod
    def __default_err():
        raise NotImplementedError()
    
    @staticmethod
    def __init_matrix(columns_):
        return pd.DataFrame(columns=columns_, index=range(len(columns_)), data=False)
    
