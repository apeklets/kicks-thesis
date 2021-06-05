# Kick Drum Generation with Neural Networks

This repository contains the code used to conduct the experiments for our bachelor thesis. 
You can read the full text [here](https://dev.wdamen.com/kicks/BachelorThesis_s1028002.pdf).
We have an accompanying site with sound examples [here](https://dev.wdamen.com/kicks/).

## Running experiments
Each of the models has its own code dependencies, so we ran each model in a different Python virtual environment. 
We used [Poetry](https://python-poetry.org/) for this, but any other environment management will work. 
The approach is to create a virtual environment in the correct folder, install the dependencies listed in `pyproject.toml` or `requirements.txt`, 
and then run `pip install -r requirements_cuda110.txt` to install the latest PyTorch version in that virtual environment.

## Progressive WaveGAN
The folder `experiments/wavegan` contains our adaptation of the WaveGAN model, based on [this PyTorch version](https://github.com/auroracramer/wavegan).

## Other models
We compared the progressive WaveGAN versions to some other models:

- [SpecGAN](https://github.com/chrisdonahue/wavegan)
- [Nistal PGAN](https://github.com/SonyCSLParis/Comparing-Representations-for-Audio-Synthesis-using-GANs)
- [WaveRNN](https://github.com/fatchord/WaveRNN)
- [WaveNet](https://github.com/golbin/WaveNet)

These all also got their own folder in `experiments/`, and we made some small changes to adapt these to our goals.

## Quality Measure
The folder `notebooks/quality` contains the source code for the quality measure we defined for evaluating kick drum sample quality. 
The folder `notebooks/` further contains some jupyter notebooks that document some informal experiments we conducted during the design of this quality measure.

## Results
The `results` folder contains some config file, samples and quality measure graphs from a couple of our experiments.
