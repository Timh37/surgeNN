# surgeNN

'surgeNN' (surge *N*eural *N*etworks) is a repository for training data-driven storm-surge models (MLR of Tadesse et al. (2020) and neural networks: for now, LSTM- and ConvLSTM models) using predictors from the ERA5 atmospheric reanalysis and de-tided tide-gauge observations from GESLA3, and producing predictions with the trained models. Scripts are also provided to evaluate and plot the performance of the data-driven models and compare them to the hydrodynamic model GTSM (Muis et al., 2020; 2023). This code repository underlies Hermans et al. (in preparation for submission to NHESS): Computing Extreme Storm Surges in Europe Using Neural Networks.

The DenseLoss code originates from: https://github.com/SteiMi/denseweight.

## Dependencies (see req.txt)
- numpy
- pandas
- xarray
- zarr
- netcdf4
- scipy
- sklearn
- tqdm
- yaml
- tensorflow
- keras
- KDEpy
- matplotlib (only for visualization)
