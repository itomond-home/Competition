# Taken from the notebo
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args
import numpy as np

# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class GroupTimeSeriesSplit(_BaseKFold):
    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_size=None
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))
        group_test_size = n_groups // n_folds
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []
            for train_group_idx in unique_groups[:group_test_start]:
                train_array_tmp = group_dict[train_group_idx]
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)
            train_end = train_array.size
            if self.max_train_size and self.max_train_size < train_end:
                train_array = train_array[train_end -
                                          self.max_train_size:train_end]
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)
            yield [int(i) for i in train_array], [int(i) for i in test_array]

from typing import Optional
from collections import OrderedDict
from lightgbm.callback import CallbackEnv
from tqdm.auto import tqdm

class LgbmProgressBarCallback:
    description: Optional[str]
    pbar: tqdm

    def __init__(self, description: Optional[str] = None):
        self.description = description
        self.pbar = tqdm()

    def __call__(self, env: CallbackEnv):

        # 初回だけProgressBarを初期化する
        is_first_iteration: bool = env.iteration == env.begin_iteration

        if is_first_iteration:
            total: int = env.end_iteration - env.begin_iteration
            self.pbar.reset(total=total)
            self.pbar.set_description(self.description, refresh=False)

        # valid_setsの評価結果を更新
        if len(env.evaluation_result_list) > 0:
            # OrderedDictにしないと表示順がバラバラになって若干見にくい
            postfix = OrderedDict(
                [
                    (f"{entry[0]}:{entry[1]}", str(entry[2]))
                    for entry in env.evaluation_result_list
                ]
            )
            self.pbar.set_postfix(ordered_dict=postfix, refresh=False)

        # 進捗を1進める
        self.pbar.update(1)
        self.pbar.refresh()

