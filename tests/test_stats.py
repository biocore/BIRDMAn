import numpy as np
import pytest

from birdman import stats


class TestHotelling:
    def test_hotelling(self):
        p, n = 50, 100  # 50 features, 100 draws
        Xrand = np.random.normal(loc=0, scale=0.1, size=(p, n))
        Xreal = np.random.normal(loc=0, scale=0.1, size=(p, n))
        Xreal[0, :] += 10  # increment first feature uniformly by 10

        _, pval = stats._hotelling(Xrand)
        assert pval > 0.01
        _, pval = stats._hotelling(Xreal)
        assert pval < 0.05

    def test_hotelling_feat_gt_samp(self):
        p, n = 102, 100  # 102 features, 100 draws
        Xrand = np.random.randn(p, n)

        with pytest.raises(ValueError) as excinfo:
            stats._hotelling(Xrand)

        exp_msg = "Number of samples must be larger than number of features!"
        assert str(excinfo.value) == exp_msg
