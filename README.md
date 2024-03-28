# Official Implementation for Sinusoidal Positional Encoding (SPE)

This is the Official Code for paper 

"Learning High-Frequency Functions Made Easy with Sinusoidal Positional Encoding". 

<!---
Hide:
Chuanhao Sun, Zhihang Yuan, Kai Xu, Luo Mai, Siddharth N, Shuo Chen, Mahesh K. Marina.
-->

(Under Review)

## Baselines

We conducted extensive use case evaluations including 1) 1-d regression, 2) 2-d speech synthesis and 3) 3-d NeRF.
The results of baseline comparison is listed in [./baselines](./baselines) folder.

### 1-D regression

We implement the 1d regression task with SPE. See the implementation of ["Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains"](https://github.com/tancik/fourier-feature-networks/tree/master).

<p align="center">
  <img src="./baselines/1d_regression/1d_loss_original.png" alt="Loss plots for 1-d regression task using original PE.">
  <br>
  <em>Fig: Loss plots for 1-d regression task using original PE.</em>
</p>

<p align="center">
  <img src="./baselines/1d_regression/1d_loss_grff.png" alt="Loss plots for 1-d regression task using Gaussian Random Fourier Features (Gaussian RFF).">
  <br>
  <em>Fig: Loss plots for 1-d regression task using Gaussian Random Fourier Features (Gaussian RFF).</em>
</p>

<p align="center">
  <img src="./baselines/1d_regression/1d_loss_ours.png" alt="Loss plots for 1-d regression task using our proposed SPE.">
  <br>
  <em>Fig: Loss plots for 1-d regression task using our proposed SPE.</em>
</p>

### Text2Speech Generation 

(Full Implementation Coming Soon)

We implement the SPE with FastSpeech, where the full connection layers perform a bottleneck when generating more details of the speech signal. See the implementation of ["FastSpeech"](https://github.com/xcmyz/FastSpeech)

![Details of Speech Generation with SPE (1)](imgs/details3.png)

The gain is reflected on different fidelity metrics as well.


![Speech Table](imgs/speech.png)


### NeRF

(Full Implementation Coming Soon)

We implement the SPE with FreeNeRF and achieve the state-of-the-art performance on few-view NeRF fidelity. See the implementation of ["FreeNeRF"](https://github.com/Jiawei-Yang/FreeNeRF)

The modification we made is illustrated in the following figure:

![SPE and NeRF](imgs/spenerf.png)

And for the first time, we managed to explain why the Basic NeRF takes 10 components on Blender dataset by learning the frequency features directly.

![Explain the Default Configuration of NeRF](imgs/hist.png)

Our model achieves SOTA performance on few-view NeRF tasks

![Basic Chair NeRF](imgs/chair3.png)

![Chair NeRF](imgs/chair2.png)

Compared with different SOTA NeRF method with different encoding methods, the SPE shows stable gain that depends on the exact implementation of NeRF method. The fidelity is shown in the following table.

![Performance NeRF](imgs/nerf.png)
