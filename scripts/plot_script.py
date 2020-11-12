import matplotlib
import pickle

from songbird2.plot import plot_differentials
from songbird2.util import collapse_results

matplotlib.use('TKAgg', warn=False, force=True)

with open("tests/data/moving_pictures_model.pickle", "rb") as f:
    load_fit = pickle.load(f)

fit = load_fit["fit"]
res = fit.extract(permuted=True)
colnames = ["C1", "C2", "C3", "C4"]

beta_df = collapse_results(res["beta"], colnames)
ax = plot_differentials(beta_df, "C1")

matplotlib.pyplot.show()
