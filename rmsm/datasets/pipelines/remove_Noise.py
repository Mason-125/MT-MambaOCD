import numpy as np

from ..builder import PIPELINES
from ..rampy import smooth

@PIPELINES.register_module()
class Smoothing(object):
    def __init__(self, method='GCVSmoothedNSpline', **kwargs):
        self.method = method
        self.kwargs = kwargs

    def __call__(self, results):
        data = []
        x = results['raman_shift']
        for i in range(len(results['labels'])):
            y = results['spectrum'][i]
            # Smooth Raman spectrum
            y_smooth = smooth(x, y, method=self.method, **self.kwargs)
            data.append(y_smooth)
        data = np.array(data)
        results['spectrum'] = data

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'method={self.method})')
        return repr_str
