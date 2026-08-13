
# Beer_dataset_MSCP_ML

R code for generating a synthetic beer fermentation dataset and illustrating a 
hybrid Multivariate Statistical Process Control (MSPC) and Machine Learning (ML) 
monitoring framework.

## Overview

The script generates a fully synthetic dataset designed to emulate realistic beer 
fermentation conditions under controlled and reproducible scenarios.

The workflow includes:

- Synthetic batch generation
- Fault injection
- PCA-based MSPC
- XGBoost classification
- SHAP interpretation
- Hybrid MSPC-ML decision framework
- Reproduction of figures and results reported in the manuscript

## Reproducibility

All data are generated synthetically within the script. No industrial or confidential datasets are required.

## Associated manuscript

This repository accompanies the manuscript: "Role and relevance of Data Analytics in beverage and brewing industry"
currently under peer review.

## Requirements

- Install [R](https://www.r-project.org/)
- Required packages are:

```
pkgs <- c("dplyr", "qcc", "xgboost", "pROC", "corrplot", "ggplot2", "patchwork", 
          "scales", "grid", "SHAPforxgboost")
pkgs <- setdiff(pkgs, installed.packages()[,"Package"])
 


```
## Authors

- Erika Francesch-Domenech
- Ana Isabel Fernandez-Calviño
- Salvador Naya-Fernández
- Javier Tarrío-Saavedra

## Citation
This code is archived on Zenodo: 
DOI: https://doi.org/10.5281/zenodo.21917783

## Usage

Run:

```r
source("case_study_code_synthetic_data.R")
```
