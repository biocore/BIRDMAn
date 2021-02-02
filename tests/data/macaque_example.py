"""
Script to generate example files
Qiita ID: 11835
"""

import biom
import pandas as pd

tbl = biom.load_table("data/macaque/56587_reference-hit.biom")
tbl_df = tbl.to_dataframe().T
num_samples, num_feats = tbl_df.shape
tbl_df.columns = [f"OTU{i+1}" for i in range(num_feats)]
md = pd.read_csv("data/macaque/11835_20181024-101425.txt", sep="\t",
                 index_col=0)

# there are a few duplicates so we'll drop them
md = md[~md.index.duplicated(keep=False)]
md.index = md.index.astype(str)
samps_to_keep = md.index.tolist()
num_samples = len(samps_to_keep)
tbl_filt_df = tbl_df.loc[samps_to_keep]

# take only features present in more than 50% of samples
feat_prevalence = tbl_filt_df.clip(upper=1).sum(axis=0)/num_samples
feats_to_keep = feat_prevalence[feat_prevalence >= 0.6].index.tolist()
tbl_filt_df = tbl_df.loc[samps_to_keep, feats_to_keep]

tbl_filt = biom.table.Table(
    tbl_filt_df.T.values,
    sample_ids=samps_to_keep,
    observation_ids=feats_to_keep
)

with biom.util.biom_open("macaque_tbl.biom", "w") as f:
    tbl_filt.to_hdf5(f, "testing")

md.to_csv("macaque_metadata.tsv", sep="\t", index=True)
