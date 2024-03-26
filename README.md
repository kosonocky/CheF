# Chemical Function (CheF) Dataset and Model

[![DOI](https://zenodo.org/badge/656230513.svg)](https://zenodo.org/badge/latestdoi/656230513)

**Mining Patents with Large Language Models Demonstrates Congruence of Functional Labels and Chemical Structures**

Clayton W. Kosonocky, Claus O. Wilke, Edward M. Marcotte, and Andrew D. Ellington

(Under review)


## Dataset

The CheF dataset contains just under 100K molecules and their ChatGPT-summariezd patent-derived functional labels.

The CheF dataset can be found in /results/CheF_100K_final.csv


## Visualizer

A visualization of the 100K molecule CheF dataset can be found at [chefdb.app](chefdb.app).

This visualization is a t-SNE projection of Daylight fingerprints obtained from RDKit and is colored based on whether or not that molecule contains a given label.

The current features of the app are as follows:
- Highlight points based on selected label. This displays PubChem CID, SMILES, associated CheF labels, and molecular structure
- Zoom in/out
- Drag plot to change location (if zoomed in)
- Toggle tooltips on only highlighted points or all points
