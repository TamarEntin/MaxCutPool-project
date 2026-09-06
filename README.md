# MaxCutPool — Reproduction & Analysis

Deep Learning final project: reproducing and analyzing **MaxCutPool** (ICLR 2025), a graph
pooling method for GNNs based on the MaxCut problem. Unlike traditional pooling methods that
group similar, closely connected nodes, MaxCutPool selects nodes that are separated and
minimally connected — useful for heterophilic graphs, where connected nodes tend to differ.

- Paper: [MaxCutPool (arXiv:2409.05100)](https://arxiv.org/abs/2409.05100)
- Original code: [NGMLGroup/MaxCutPool](https://github.com/NGMLGroup/MaxCutPool)

**Submitters:**
- Tamar Entin — 210014411
- David Poss — 316936111
- Zohar Ben Hayoun — 315571604

## What's in this repo

```
notebooks/
└── MaxCutPool_Project.ipynb   # the project notebook — start here
MaxCutPool/
├── source/                    # model / layers / data code
├── config/                    # Hydra configs (datasets, poolers, architectures)
├── run_maxcut.py              # MaxCut optimization (ring graph)
├── run_graph_classification.py# graph classification (MUTAG, NCI1, ...)
├── run_node_classification.py # node classification (Roman-Empire)
├── requirements.txt / environment.yml
└── README.md, LICENSE         # from the original MaxCutPool repo
```

`MaxCutPool/` is a copy of the original repo's code with one fix applied to the three
`run_*.py` scripts: the training accelerator now falls back to CPU automatically
(`accelerator='gpu' if torch.cuda.is_available() else 'cpu'`) when no GPU is available.

Only the files needed to run the notebook are included — generated artifacts (`data/`,
`logs/`, `multirun/`, checkpoints) are left out and are recreated automatically the first
time each experiment runs.

## The notebook

`notebooks/MaxCutPool_Project.ipynb` reproduces the paper's three core experiments —
direct MaxCut optimization, graph classification, and node classification — plus extra
experiments on hyperparameter sensitivity and training stability. It expects to sit next
to `MaxCutPool/` exactly as laid out in this repo (it resolves the code directory as
`../MaxCutPool` relative to the notebook, and `cd`s into it before running anything).

## Setup

```bash
conda env create -f MaxCutPool/environment.yml
conda activate maxcutpool
```

(or install `MaxCutPool/requirements.txt` into an existing PyTorch + PyTorch Geometric
environment). Then launch Jupyter and run `notebooks/MaxCutPool_Project.ipynb` top to
bottom. Datasets are downloaded automatically on first use; a GPU is recommended but not
required.
