# DRSA-AUDIO
Disentangled explanations for neural network predictions on audio data.

## Description
- Code to train a CNN to perform music genre classification on the GTZAN dataset [[1]](#1)
- Relevance attribution with Layerwise-Relevance Propagation (LRP) [[2]](#2)
- Implementation of Disentangled Relevant Subspace Analysis (DRSA) [[3]](#3) to decompose the standard explanation into semantically rich sub-explanations that reassemble the standard explanation

Results are provided at [https://sharkhai/github.io/drsa-audio-results/](https://sharckhai.github.io/drsa-audio-results/)

## Further work
- Notebooks will follow

## Installation
- Install required packages with

```bash
pip install -r requirements.txt
```

- Install setup.py in editable mode

```bash
pip install -e setup.py
```




## References
<a id="1">[1]</a> 
G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," in IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293-302, July 2002, doi: 10.1109/TSA.2002.800560.

<a id="2">[2]</a> 
Sebastian Bach et al. “On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation”. In: PLOS ONE 10.7 (July 2015), pp. 1–46. doi: 10.1371/journal.pone.0130140. 

<a id="3">[3]</a> 
Pattarawat Chormai et al. “Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2024), pp. 1–18. doi: 10.1109/tpami.2024.3388275.
