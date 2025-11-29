# Perturb-OT
Cross-modality matching and prediction of **perturb**ation response with labeled Gromov-Wasserstein **O**ptimal **T**ransport.

<img src="imgs/schematic.png" alt="Schematic" width="700"/>

## Repository layout
```
├── .gitignore
├── README.md
├── perturbot
│   ├── cv      # Cross-validation experiments
│   └── perturbot
│       ├── match
│       │    ├── ot_labels.py   # Label-constrained Entropic Optimal Transport
│       │    ├── ott_egwl.py    # Label-constrained Entropic Gromov-Wasserstein
│       │    ├── cot_labels.py  # Label-constrained Co-OT (Redko et al., 2020)
│       │    └── cot_feature.py # Feature-feature OT based on Co-OT concept
│       ├── predict
│       └── eval
├── scvi-tools              # forked `scvi-tools` with label adaptation
└── ott                     # forked `ott` with label adaptation

```

## Installation
`perturbot/` uses the modified `scvi-tools` and `ott` submodules which can be installed with `pip install`.
```bash
cd scvi-tools/
pip install .
cd ../ott/
pip install .
cd ../perturbot
pip install .
```
## Patch applied
1. Execute the commands in the Installation section of the repo

2. At that point `perturbot` is almost empty (only __init__ and utils), to fix that, go into Perturb-OT/perturbot/perturbot, copy all submodules and paste them inside the Perturb-OT/perturbot/build/lib/perturbot folder.
```bash
cp -r perturbot/perturbot/. perturbot/build/lib/perturbot/
```

3. Now uninstall and reininstall `perturbot` specifically:
```bash
pip uninstall perturbot
cd perturbot/
pip install .
```

At this point `perturbot.match` **should work** when imported. **However** you'll likely hit dependency issues regarding **jax and anndata**:

### Fixing `jaxlib.xla_extension` error
1. What's happening is some good old mismatch between the code in some depedencies and the updated API of JAX 0.8.0
```bash
pip install "jax[cpu]==0.4.36" "jax[cuda]==0.4.36" "jaxlib==0.4.36"
```

That should solve it.

### Fixing `anndata` errors
1. The importing dynamics of anndata slightly changed in recent versions which makes it no longer possible to import read at the very top level so to keep dependency code as is we need to downgrade to an anndata version that supported it:
```bash
pip install "anndata==0.10.9"
```

**That should be it ! Both perturbot.match and perturbot.predict should be importable now :D !**

## Usage
```python
import numpy as np
from sklearn.decomposition import PCA
from perturbot.match import (
    get_coupling_cotl, 
    get_coupling_cotl_sinkhorn, 
    get_coupling_egw_labels_ott,
    get_coupling_egw_all_ott,
    get_coupling_eot_ott,
    get_coupling_leot_ott,
    get_coupling_egw_ott,
    get_coupling_cot, 
    get_coupling_cot_sinkhorn, 
    get_coupling_gw_labels,
    get_coupling_fot,
)
from perturbot.predict import train_mlp

# Generate data
n_samples = 300
labels = [0,1,2,3]
X_dict = {k: np.random.rand(n_samples,1000) for k in labels}
Y_dict = {k: np.random.rand(n_samples,2000) for k in labels}
pca = PCA(n_components=50)
X_reduced = {k: pca.fit_transform(X_dict[k]) for k in labels}
Y_reduced = {k: pca.fit_transform(Y_dict[k]) for k in labels}


# Learn matching in the latent space
T_dict, log = get_coupling_egw_labels_ott((X_reduced, Y_reduced)) # Other get_coupling_X methods be used

# Train MLP based on matching
model, pred_log = train_mlp((X_dict, Y_dict), T_dict)

# Learn feature-feature matching
T_feature, fm_log = get_coupling_fot((X_dict, Y_dict), T_dict)
```
See [documentation](https://genentech.github.io/Perturb-OT/) and [manuscript](https://arxiv.org/abs/2405.00838) for more details.
## Support
Please submit issues or reach out to jayoung_ryu@g.harvard.edu.

## Authors and acknowledgment
Jayoung Ryu, Romain Lopez, & Charlotte Bunne

## Citation
If you have used Perturb-OT for your work, please cite:
```
@misc{ryu2024crossmodality,
      title={Cross-modality Matching and Prediction of Perturbation Responses with Labeled Gromov-Wasserstein Optimal Transport}, 
      author={Jayoung Ryu and Romain Lopez and Charlotte Bunne and Aviv Regev},
      year={2024},
      eprint={2405.00838},
      archivePrefix={arXiv},
      primaryClass={q-bio.GN}
}
```
