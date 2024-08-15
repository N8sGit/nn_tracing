# NN_Staining

## Table of Contents
- [Overview](#overview)
- [Concepts](#concepts)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)
- [Note on terminology](#terminology)
## Overview

nn\_staining is a experiment in explainable AI that aims to improve model interperability. The basic pitch is this: most approaches to explainable AI involves tracking and representing metrics *about* the model. The goal here is to track and represent metrics *of* the model *itself*. 

Essentially the goal is to construct graphs of the internal dynamics of the model itself during training, to “banish the black box”. You can think of it sort of like a brain scan. 

We also capture various data points about the model's states throughout the training lifecycle, such as the activation histories, bias terms, and weights. Why? That part I haven't figured out yet :) but the reasoning is that there may be hidden patterns or clues about how the model evolves during training in these numbers yet to be discovered. Typically they are just discarded, and never even looked at, but I want to at least want to experiment with analyzing this data. 

The idea is that neural networks are *evolutionary systems*, that is to say, they gain order over time. What if we could find ways to chart that order? 

What this project aims to do is to take a tour through an artificial neural network model, taking snapshots of it from the inside as you go. Later those snapshots are analyzed into “breadcrumbs” and reconstructed into visualizations representing areas of statistical interest about the model. 

The ultimate aspiration of the project is to develop a method to identify meaningful internal structures in the model associated with its final outputs, so that we can know precisely what parameters and neurons contributed to that outcome in exactly what ways. 

Once we can isolate these structures, we can *in theory* insert programmatic hooks into them, to alter, delete, or otherwise control the model in ways previously thought impossible. That, at least, is the long term vision.

## Concepts 

For a deeper theoretical exposition , consult the white paper (forthcoming). Here I go into more detail about the static big picture parts of the project, its methods, and motivations. This readme will be more about documenting fast-changing implementation details about the code. 

Nevertheless, here's a basic high level conceptual overview. 

To give you a basic idea what this project is about, let me take you on a brief journey on what went through my head to serve as the genesis of it. 

I was curious why neural network models do what they do and come to the decisions they make. I was annoyed by their reputation a "black boxes", and feel that, until we can gain more transparency into how these systems work from the inside out we will never have full control over them and they will always be somewhat of a liability. It also irritates me how from our perspective, when we look at what's going on in these models we just see streams of meaningless numbers. But from the model's perspective, it's able to interpret those numbers meaningfully. It feels wrong that there is no interface between human and machine to bridge this gap of comprehension. 

Then I realized: *the only reason NNs are black boxes is because we don't record the paths data take through it*. Imagine if we followed a particular signal of data through the model, drawing a line to trace its path through the network. Do this enough times, and we we will eventually highlight the pathways of most consequence through the network, or what I call structures. 

Then we do could do something like color code these distinct pathways of regular tendency, and associate them with final outputs, and that's how I got the idea of "staining". 

In my first attempts to implement this idea, I tried out something like a "probe" that recorded the state of every neuron during the entire training cycle. I quickly found myself lost in a sea of numbers without any clues as to how to find hints of organization in it. 

That's when I realized: to capture what changes in a model, first we must capture *what doesn't change*

In essence it all revolves around what I call the *network trace*

- **Network Trace**

What is a network trace? Basically it's . When I was studying how neural networks get trained in libraries such as TensorFlow and PyTorch, I realized that they had a very peculiar geometry. They are, in essence, hypercubes, 4 dimensional cubes, where-- 
- the 4th dimension is time (captured by epochs)
- the 3rd dimension is the network layers 
- The 2nd dimension is stacks of neurons in the layers 
- And the 1st dimension is the individual neurons themselves

A network trace is simply a hierarchically nested data structure that captures these uniformities throughout the training process. Every Epoch, Layer, and Neuron gets a unique id, and when we want to examine, record, or look it up we treat the network trace as a hash table using a *network signature* to track it down. So if I want to see what happened to neuron # 20, located at hidden layer 1, at epoch 30, I can find it like such``my_neuron = model.network_trace['E_30']['L_hidden_1']['n_20']``

Later, we import this hierarchical structure into a jupyter notebook, convert it into a multi-indexed pandas dataframe, and now we have a neat, tidy representation of the whole model and everything we cared to capture about it as it went through its training. Then comes the hard part.

- **Meta-modeling**
The next thing that comes to mind is to how to actually analyze the model's metadata. 


## Usage


Open ``analysis.ipynb`` and run the first cell. This will run the main process in ``main.py`` and load the network trace into the jupyter notebook for analysis. 

Alternatively, in the terminal run:
``
python main.py
`` to execute the main script without analysis. This will execute the tracing logic and train the model and log various details in the terminal. This can be helpful for debugging.  

For now image files will be ouput to ``/results``. Future versions of the project may include a better UX for this and other operations. 

See [Roadmap](#roadmap) for more information about the project's current and hoped for eventual capabilities.

The developer can modify various settings in 
``
python main.py
`` to control the behavior of the core ``Network_Trace`` class. Such as:

- ``epoch_intervals``: Types: ``[Optional]`` ``int``, ``List[int]``, defaults to ``None``. Controls the frequency per epoch at which the scanning logic is applied during model training. If an ``int`` is provided, scans will occur at every `int`th interval. If ``List[int]`` is provided, scans will occur only at those specified epochs. If no value is provided, scans will occur during every epoch. **Note: Not Recommended for large models**: 
- `drop_batches`: Types: ``[Optional]`` ``bool`` defaults to ``True``. If set to ``True`` intermediate metadata will be dropped after computing final metrics. This is to help reduce storage footprint. Otherwise, intermediate metadata will be conserved across the training lifecycle. 

In the future I hope to add a UI to make configuration more accessible. 


... TBC

## Modifying the NN model
While the ideal goal of this project is to have an abstract specification that works with any model, the reality at the moment is that the code in its current state is probably closely coupled to the exact parameters of the SimpleNN class and will break if exposed to anything too fundamentally different in structure. Once I get a good proof of concept, hopefully I will figure out how to make this work for all types of models. 


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
- Continue to research and improve on the mathematical analysis used to select data
- Generalize into an implementation-agnostic abstract specification that works with as many model types as possible 
- Improve usage (either by providing better scripting logic so sourcefiles don't need to be manually modified and can be set via terminal commands, or by providing a GUI )
- Add benchmarks to measure overhead

## Terminology: 
The term "staining" came to mind when I first thought of this idea. It is meant to invoke the tissue staining methods used in histology by biologists studying cells. They add certain biochemical markers to tissues to detect various cellular differences. Since this project tries to isolate and color code different structures in artificial neural networks, the metaphor felt apt. I've grown less attached to the image over time, but until I think of something more descriptive I'll keep it for legacy purpose. 
