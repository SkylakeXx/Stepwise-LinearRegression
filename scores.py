# sample scores
from sklearn.metrics import make_scorer
from functools import partial
import numpy as np

scorer_down_is_better = partial(make_scorer, greater_is_better=False)
scorer_up_is_better = partial(make_scorer, greater_is_better=True)

@scorer_down_is_better
def mean_squared_error(y_true, y_pred, **kwargs):
    return  np.mean(np.square(y_true-y_pred))
