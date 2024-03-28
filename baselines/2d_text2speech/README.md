# [Submission] Learning High-Frequency Functions Made Easy with Sinusoidal Positional Encoding

Notice: This part is still under restructuring for an easier implementation.

## Experiments Implementation

### Ground Truth

For illustration purpose, we will use the Text to Speech Model (["FastSpeech"](https://github.com/xcmyz/FastSpeech)) to generate the sentence `The crime soon became public.`. The corresponding audio file is LJ013-0031 from ["LJSpeech dataset"](https://keithito.com/LJ-Speech-Dataset/).

We have included the ground truth video files, as long as the other three synthetic audio file within the directory [./experiments_results/](./experiments_results).

The corresponding spectrogram plot can be found in the same directory, plotted by the jupyter notebook script [./mel-spectrogram.ipynb](./mel-spectrogram.ipynb).

### Baseline Experiments

We modified the FastSpeech implementation in order to test the performance among FastSpeech without PE, FastSpeech with PE and FastSpeech with SPE. We have use `NOTE` keyword within the [./model.py](./model.py) to highlight the changes we made to implement SPE.

- For FastSpeech without PE, we follow their implementation of FastSPeech.
- For FastSpeech with PE, we add single one PE layer after the decoder, and before the linear layer below the decoder. (See forward pass within the `model.py` file).
- For FastSpeech with PE, we add SPE layer (PE + Linear + SIREN) after the decoder, and before the linear layer below the decoder. (See forward pass within the `model.py` file).

To make sure all experiments are comparable, we use `L=5` for both PE and SPE. The reason to choose `L=5` is that it can achieve optimal performance for PE within our search range from 1 to 10.
