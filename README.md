# Project Title

A brief description of what this project does and its purpose.

## Table of Contents
- [Overview](#overview)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)

## Overview

nn\_staining is a experiment in explainable AI that aims to improve model interperability. The basic pitch is this: most approaches to explainable AI involves tracking and representing metrics *about* the model, the goal here is to track and represent metrics the model *itself*. 

Essentially the goal is to construct graphical representations of the internal dynamics of the model itself during training, to “banish the black box”. You can think of it like a brain scan. 

What this project aims to do is to take a tour through an artificial neural network model, taking snapshots of it from the inside as you go. Later those snapshots are analyzed into “breadcrumbs” and reconstructed into visualizations. 

The ultimate goal of the project is to develop a method to identify meaningful internal structures in the model associated with its final outputs, so that we can know precisely what parameters and neurons contributed to that outcome in exactly what ways. Once we can isolate these structures, we can in theory insert programmatic hooks into them, to alter, delete, or otherwise control the model in ways previously thought impossible. 

For a deeper theoretical exposition , consult the white paper (forthcoming). Here I go into more detail about the project, its methods, and motivations. 

## Usage
In the terminal run:
``
python main.py
`` to execute the main script.

For now image files will be ouput to ``/results``. Future versions of the project may include a better UX for this and other operations. 

See [Roadmap](#roadmap) for more information about the project's current and hoped for eventual capabilities.

The developer can modify various settings in 
``
python main.py
`` to control the behavior of the core ``Network_Trace`` class. Such as:

- ``epoch_intervals``: Types: [Optional] ``int``, ``List[int]``, defaults to ``None``. Controls the frequency per epoch at which the scanning logic is applied during model training. If an ``int`` is provided, scans will occur at every nth interval. If ``List[int]`` is provided, scans will occur only at those specified intervals. If no value is provided, scans will occur during every epoch. **Note: Not Recommended for large models**: 
- `drop_batches`: Types: [Optional] ``bool`` defaults to ``True``. If set to ``False`` intermediate metadata will be dropped after computing final metrics. This is to help reduce storage footprint. 


... TBC



## Dependencies

For a list of required modules, see:
``
python requirements.txt 
``

## Roadmap

Currently the project is at the proof of concept stage. The minimum viable proof of concept is defined as the capability to scan a simple binary classification model to produce simple visualizations about its most salient inner workings. 

- Proof of Concept (Current)
- Draft of white paper 
- Add support for multi-classification ANNs
- Generalize into an implementation-agnostic abstract specification that works with as many model types as possible 
- Improve usage (either by providing better scripting logic so sourcefiles don't need to be manually modified and can be set via terminal commands, or by providing a GUI )
- Add benchmarks to measure overhead 