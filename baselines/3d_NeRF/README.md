# [Submission] Learning High-Frequency Functions Made Easy with Sinusoidal Positional Encoding

Notice: This part is still under restructuring for an easier implementation.

## Implementation of Following Methods

Please check 'run_nerf.py' and 'run_nerf_helpers.py' for all supported experiments.

For the rest experiment with InstantNeRF, NeRFfacto, etc. Please refer to the implementation in [NeRF Studio](https://docs.nerf.studio/). We will merge those methods into this repo later on.


### Baselines

NeRF, DietNeRF, FreeNeRF, SIREN NeRF - APE (Based on APE Paper)

### Our Approaches

**By default we use FreeNeRF (based on DietNeRF) as our base method.** 

FreeNeRF + SPE, FreeNeRF + SPE + Vanilla GAN, FreeNeRF + SPE + WGAN, 

**There are also few experiments that are not included in the paper but probably worth more investigation.**

FreeNeRF + Focalloss, FreeNeRF + Residual SPE

## Usage

All implementations are based on DietNeRF, also refer the official implementation FreeNeRF. Please following these instructions to setup the expeiments.

The core scripts with SPE are structured in the same style as DietNeRF and FreeNeRF, should work seamlessly with FreeNeRF etc.

DietNeRF: [DietNeRF implementation](https://github.com/ajayjain/DietNeRF)

FreeNeRF: [FreeNeRF implementation](https://github.com/Jiawei-Yang/FreeNeRF)

