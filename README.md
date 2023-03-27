# SYMBOLIC MUSIC STRUCTURE ANALYSIS

This repository contains the code to replicate the results of the paper: [Symbolic Music Structure Analysis with Graph Representations and Changepoint Detection Methods](https://arxiv.org/abs/2303.13881).


## Requirements

```
musicaiz==0.1.0
```

## Files in this repository

`bps_midi.py`: converts the BPS notes.csv files in MIDI files.

`graph_figure1.py`: prepares a figure with a BPS file in which the novelty cursve of the graph adjajency matrix is shown with structure annotations and predictions per level.

`figure_ssm.py`: prepares a figure with a BPS file in which the boundary candidates and SSM with novelty curve are shown for Norm algorithm.

`dataset_analysis.py`: gets the analytics of the SWD and BPS datasets.

Code for reproducibility:
- `graph_bps.py`
- `norm_bps.py`
- `graph_swd.py`
- `norm_swd.py`

## Authors

- Carlos Hernandez-Olivan
- Sonia Rubio Llamas
