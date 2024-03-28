The implementation of 1D regression follows closely the original approach described in the paper ["Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"](https://github.com/tancik/fourier-feature-networks/tree/master).


We have modified the first Dense layer, replacing it with a Sinusoidal layer, to align with our proposed SPE methodology.
(For details, refer to the `make_network` function in [1d_regression_original.ipynb](./1d_regression_original.ipynb) and
[1d_regression_ours.ipynb](./1d_regression_ours.ipynb)
).
Moreover, we also included results for Gaussian Random Frourier Features for 1-d regression. The implementation can be found in [1d_regression_grff.ipynb](./1d_regression_grff.ipynb)

Additionally, the loss plots for

- the Fourier features network and,
- the Gaussian Random Fourier Features and,
- the SPE version

can be found in the same directory.