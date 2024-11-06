# DRSA-AUDIO

Disentangled explanations for neural network predictions on audio data. [GitHub Webpage](https://sharckhai.github.io/drsa-audio-results/)

## Description

This repo provides code to extract concept-based explanations with Disentangled Relevant Subspace Analysis (DRSA) [[1]](#1) on Convolutional Neural Networks (CNNs). Experiments were conducted within 
the scope of my master's thesis at the Machine Leanring Group TU Berlin on CNNs trained on log-mel-spectrograms (i.e., 2D audio representations).
Extracting explanations on a concept-level provides deeper insights into the models descicion behaviour and the data domain. Further information about the 
research methodology and process, as well as the results of this work are provided on the webpage which is linked above. On that page you will also find a link to the thesis report.

This repo contains the following components:

- Code to construct and train a CNN to perform audio classification, e.g, music genre classification, with a ready-to-use well-considered preprocessing pipeline.
- Framework for relevance attribution with Layerwise-Relevance Propagation (LRP) [[2]](#2) using the [Zennit](https://github.com/chr5tphr/zennit) [[3]](#3) library.
- An implementation of DRSA to decompose the standard explanation (local explanation) into semantically rich sub-explanations 
representing different class specific objects that guide a the models descision.

For a more extensive overview of the experiments undertaken with this code please refer to the webpage linked above.

## Installation
We use ```python>=3.10```. Navigate to the root of the package and execute the following commands.
- Install required packages with

```bash
pip install -r requirements.txt
```

- Install setup.py in editable mode

```bash
pip install -e .
```

## Further work
- Notebooks will follow


## References
<a id="1">[1]</a> 
Pattarawat Chormai et al. “Disentangled Explanations of Neural Network Predictions by Finding Relevant Subspaces”. In: IEEE Transactions on Pattern Analysis and Machine Intelligence (2024), pp. 1–18. doi: 10.1109/tpami.2024.3388275.

<a id="2">[2]</a> 
Sebastian Bach et al. “On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation”. In: PLOS ONE 10.7 (July 2015), pp. 1–46. doi: 10.1371/journal.pone.0130140. 

<a id="3">[3]</a> 
Christopher J. Anders et al. “Software for Dataset-wide XAI: From Local Explana- tions to Global Insights with Zennit, CoRelAy, and ViRelAy”. In: arXiv:2106.13200 (2021).
