# Copyright (c) OpenMMLab. All rights reserved.
import random

import numpy as np
from rampy import normalise

from ..builder import PIPELINES
import pandas as pd

from ..rampy import load_txt

@PIPELINES.register_module()
class LoadDataFromFile(object):
    def __init__(self, id='ID', labels=['labels'], file_path=None, data_size=None):
        self.file_path = file_path
        self.data_size = data_size
        self.labels = labels
        self.id = id

    def __call__(self, results):
        if self.file_path is None:
            if isinstance(results, str):
                self.file_path = results
            else:
                self.file_path = results['raman_path']

        if self.data_size is None:
            self.data_size = results['data_size']

        if self.data_size is not None:
            assert len(self.data_size) == 2

        data = load_csv(self.id, self.labels, self.file_path, self.data_size)
        results = {'raman_id': data[0], 'labels': data[1], 'classes': data[2], 'raman_shift': data[3],
                   'spectrum': data[4]}

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'file_path={self.file_path}, '
                    f'data_size={self.data_size})')
        return repr_str


def load_csv(id, labels, path, data_size):
    base_dir = '../../' + path

    all_df = pd.read_csv(base_dir)

    start = 67
    end = 1009
    # ID
    raman_id = all_df[id].iloc[1:].values
    # spectrum labels
    spectrum_labels = all_df[labels].iloc[1:].values
    all_df.drop(labels, axis=1, inplace=True)
    all_df.drop(id, axis=1, inplace=True)

    # raman_shift
    raman_shift = all_df.iloc[0:1, start+1:end+1].values
    raman_shift = raman_shift.flatten()

    # raman_type
    raman_type = all_df['raman_type'].drop_duplicates().values

    # spectrum
    spectrum = all_df.iloc[1:, start+1:end+1].values

    # from sklearn.cross_decomposition import PLSRegression
    # output_shape = 900
    # pls = PLSRegression(n_components=output_shape)
    # pls.fit(spectrum, spectrum_labels)
    # spectrum = pls.transform(spectrum)
    import joblib
    # joblib.dump(pls, 'pls_model/save.joblib')

    is_test = True
    if is_test:
        # Load the PLS model from the file
        pls_model = joblib.load('pls_model/save.joblib')
        spectrum = pls_model.transform(spectrum)


    results = [raman_id, spectrum_labels, raman_type, raman_shift, spectrum]

    return results
