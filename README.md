# ANN_Tracing
![Image of a artificial neural network over graph paper](/assets/ann_graph_paper.jpg)

## Table of Contents
- [Overview](#overview)
- [Inspiration & Rationale](#inspiration--rationale)
- [API Reference](#api-reference)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Files](#files)
- [Roadmap](#roadmap)
- [Interesting Developments](#developments)
## Overview

If you are looking for more technical details about the API and how to navigate, use, and run the code, jump to the **API Reference, Usage, Dependencies, & Files** sections. For a more high level exposition on the project and its motivations and aspirations, read on.

For presentation I gave on this project, check out [https://www.youtube.com/watch?v=vsOXrBCX7h4](https://www.youtube.com/watch?v=vsOXrBCX7h4)

#### What is this?

nn_tracing (working title) is a experiment in explainable AI that aims to improve model interperability through model transparency. The basic pitch is this: most approaches to explainable AI involves tracking and representing metrics *about* the model. The goal here is to track and represent metrics *of* the model *itself*. 

Essentially the goal is to graph the internal dynamics of the model, to "light up the black box” by mapping structure to function. You can think of it sort of like a brain scan. This project aims to challenge the conventional wisdom that deep learning neural networks are impenetrable and that it's not even worth trying to crack open their interal workings.  

In practice all this comes down to is fitting all the parameters of a model into a data schema, that I call the ``network_trace``, which serves a few critical purposes:
1. Represent the model's key metrics in a form that parallels the training lifecycle, to capture the where and when of different states
2. Provide an efficient lookup table to read and write from this state dictionary
3. Serve as the basis for a pandas data-frame to conduct analysis of the model. 

What this project aims to do is to take a tour through an artificial neural network model, taking snapshots of it from the inside as you go. Later those snapshots are analyzed into “breadcrumbs” and reconstructed into visualizations representing areas of statistical interest about the model. 

The ultimate aspiration of the project is to develop a method to identify meaningful internal structures in the model associated with its final outputs, so that we can know precisely what parameters and neurons contributed to that outcome in exactly what ways. 

The next step is to reflexively apply machine learning algorithims to the model's metadata, to identify meaningful categories of data and create something like a "mental map" of the model's internal concept of its output. (If this sounds far out, read on)

## Inspiration & Rationale

I have an interest in neuroscience, and believe that there should be more interdisciplinary cross-talk between machine learning experts and brain experts.

While brain scans are a commonplace in neurology and related fields, the deep learning community rarely thinks of ANNs and similar models in terms of wanting to catalog their internal structure and function.

While I was working on this project, I came across this article [The brain as we've never seen it - Harvard Gazette](https://news.harvard.edu/gazette/story/2024/05/the-brain-as-weve-never-seen-it/). In a 10 year long collaboration between Harvard and Google, researchers applied machine learning techniques to a cubic millimeter slice of brain tissue. The result is the most detailed computer images of brain cells to date. 

If scientists are making great strides mapping the brain using machine learning, why can't we use machine learning to map artificial neural networks too? To apply machine learning as it were *reflectively* on the products of machine learning?

This idea should make intutive sense. While the metadata generated by deep learning processess is hard for us to interpret, it should be ready-made for machine learning algorithms. 

There are several ways IRL neural networks and artificial ones differ. At a certain point comparing them is like apples to oranges. In some ways, IRL NNs are vastly more complex than ANNs. The tiny cubic millimeter sample used in the Google/Harvard study, is no bigger than the period that ends this sentence yet contains an estimated 150 million synapses (micro-connections). 

However, IRL NNs have the benefit of being physical. As such, there is something actually *there* to map. ANNs decompose into a mess of numbers, and there is no explicit order to them. 

The computer has no concept of a "network", all it "sees" are numbers. There is no geometry to the system that we can trace out. So any framework that we would want to map of an ANN (or similar) has to be imposed onto those numbers and carefully deduced from them.

The good thing is, we know *for a fact* that the numbers of a model are *not random*, despite at first glance appearing meaningless. If the model can perform tasks or make predictions better than chance, it stands to reason that there are indeed patterns in those numbers corresponding to its functional behaviors. If it were completely random, its outputs would be chaotic and unusable. 

So we can at least be assured that there are indeed *patterns to discover* in that mess of numbers, if we can find the right framework to bring them out.

### Why bother?

Some machine learning enthusiasts may question the value of attempting to probe the black box at all. In my opinion the conventional thinking on the subject is too small. The value of improving model transparency may not be immediately apparent.

 But in general, there are probably at least 4 main reasons for disinterest in model transparency.

1. The focus has been on getting deep learning models to do interesting things. There is no point trying to figure out how they come to their conclusions before they produce interesting results. Dedicating resources to model transparency before getting them to do anything interesting puts it backwards. Model transparency is clearly a later step to come after we've done the "real work" of getting models to solve interesting problems.
2. Much can be inferred only by looking at I/O relationships. A grasp on throughputs is not strictly necessary and does not prohibit work on NNs
3. Even simple deep learning models produce extremely large quantities of data as side-effects of training and inference, and moreover, that data is not at all designed for human-readability. Indeed, it's rather hostile to it.
4. It wouldn't be worth the trouble.

Reasons 1-3 are defensible disincentives, but 4 is not. In programming, anything you can identify you can reference, and anything you can reference you can  directly manipulate. If we can identify meaningful statistical structures in deep learning models, this could, potentially, open a whole new world of possibilities for them. 

Conceivable benefits might include:

- More efficient knowledge transfer between models. By isolating sets of numbers that statistically correspond to certain behaviors, it might be possible to catalogue those skills in a more organized fashion.
- More efficient pruning of non-contributing neurons. 
- AI safety. Language models that are trained on massive corpora may learn and retain dangerous information. Typically, at that point, it's difficult to fully erase or overwrite its forbidden knowledge. Isolation techniques could potentially allow for the pinpointing of problem areas in a model.
- Direct, programmatic intervention in model behavior. This is probably the prospect that is most speculative, but to me also the most enticing. If for the sake of argument we discover that a model tends to behave a certain way when the value of a given weight is +1, we could in theory programmatically add that proverbial +1 ourselves, directly involving more traditional programming in model development. The dream here would be to give ourselves more direct control over how these models behave. This, again, is purely speculative. 

The point is it's narrow-minded to not even try. Who knows where it could lead?


## (Update: NEW AND IMPROVED) API Reference
See the docstrings of each module for more detailed information. 

### TraceableNN 

``TraceableNN`` is a wrapper for a model that takes hyperparameters and decorates it with recording logic. For all other purposes it behaves like a regular ANN you might define in pytorch. 

The implementation of TracableNN is planned to change, with the intent for it to eventually become more abstracted. The ultimate vision for it is that you pass a recipe for any model into it and the tracing logic gets injected and wrapped around any kind of model. 

### NetworkTrace

The core of this library. It is effectively a large lookup table with a collection of logic for writing down a bunch of numbers at static, revistable locations. 

 While I was studying how neural networks get trained in libraries such as TensorFlow and PyTorch, I realized that they had a very peculiar geometry. They are, in essence, hypercubes, 4 dimensional cubes, where-- 
- the 4th dimension is time (captured by epochs)
- the 3rd dimension is the network layers together, the whole model
- The 2nd dimension is stacks of neurons in the layers, or an individual layer  
- The 1st dimension is the rows of neurons in a layer
- The 0th dimension is the individual neuron "points" in space

A network trace is simply a hierarchically nested data structure that captures these uniformities throughout the training process. Every Epoch, Layer, and Neuron gets a unique id, and when we want to examine, record, or look it up we treat the network trace as a hash table using a *network signature* to track it down. So if I want to see what happened to neuron # 20, located at hidden layer 1, at epoch 30, I can find it like such``my_neuron = model.network_trace['E_30']['L_hidden_1']['n_20']``

Later, we import this hierarchical structure into a jupyter notebook, convert it into a multi-indexed pandas dataframe, and now we have a neat, tidy representation of the whole model and everything we cared to capture about it as it went through its training.

### ModelConfig and ModelConfigurator

* ``ModelConfig ``: A dataclass that captures a model's parameter configuration with the following schema: 
 ```bash
 input_size: int
    hidden_size: int
    output_size: int
    num_samples: int
    num_time_steps: int
    time_step_interval: List[int]
    layer_names: Dict[str, str]
    batch_size: Optional[int] = 16
    inference_batch_size: Optional[int] = 30
    data: Optional[Dict] = field(default_factory=dict)
```
* ``ModelConfigurator ``: A factory function that groups commonly associated model recipes together for loss functions, activation functions and label formats. See docstring for more details

### DataHandler
A factory for a data model. Helps to generate a synthetic data set or load an existing one for the pipeline and sets up data loaders. See docstrings for more details.


### TracablePipeline
A pipeline that combines model configs, the model configurator, the data handler, and the tracable model into a single convenient flow. If you follow the existing code it shows how to set everything up with the "iris" data set. Follow this pattern for any other dataset or model you might want to try.



## Analysis
Everything comes together in ``analysis.ipynb``.  This is where the traced model is outputed and prepared for analysis

- **Meta-modeling**
The next thing that comes to mind is to how to actually analyze the model's metadata using machine learning and other types of analysis.

Currently, t-SNE is used to cluster neurons by similarity. This is a rather crude first attempt, and I hope to involve more sophisticated techniques in future.

There are MANY different ways we can slice it here. However, there is a real risk of *apophenia*: the phenomenon of seeing patterns that are not there e.g. faces in the clouds. 

Since we don't really know what a meaningful pattern looks like, it's hard to differentiate a successful experiment from a failed one. But I do have some general avenues of research I would like to explore:

- Probablistic entropy models: e.g Hidden Markov. Probability is a proven tool for making sense of otherwise unmanagable data sets. The driving hypothesis is that when the network decides on a given classification result or prediction, certain of its elements should be in a less random state compared to baseline. If we can define those non-random boundaries, and develop a predictive or classifcatory model for differentiating or separating that relative clump of order from the general blob of randomness, that might be as good as anything we could come up with.

TBC... 


## Usage


Open ``analysis.ipynb`` and run the first few cells. This will run the main process in ``main.py`` and load the network trace into the jupyter notebook for analysis. 

NOTE: ``analysis.ipynb`` will generate large outputs. It's recommended to CLEAR ALL OUTPUTS prior to making any git commits. 

Alternatively, in the terminal run:
``
python main.py
`` to execute the main script without analysis. This will train the model, run the tracing logic, and log various details in the terminal. This can be helpful for debugging or working with the lower level code.  

Models and network traces will be saved to  ``/outputs``. Despite ``Network_Trace`` being placed on ``TracableNN``'s class, it has to be exported separately in a pickle and then "reattached" when the model is re-hydrated and run in evaluation mode in ``analysis.ipynb``. I'm not sure why this is the case, but I suspect that ``torch.save`` only respects properties native to pytorch and ignores "extraneous" additions to the model class. This should not be a problem since network traces are read-only and designed to have zero side-effects or influence on training or inference.

See [Roadmap](#roadmap) for more information about the project's current and hoped for eventual capabilities.

In the future I hope to add a UI to make configuration more accessible. 

## Dependencies

For a list of required modules, see:
``
python requirements.txt 
``
A few libraries not listed in the requirements.txt
- ``pandas                    2.2.2 ``
- ``adjusttext                1.2.0 ``
- ``plotly                    5.9.0  ``

## Files

- ``data.py``: This is where you would set up the data for the model.

- ``main.py``: the main process. Everything comes together here. Currently this is simply where the pipeline gets executed. Results are then saved away to ``/outputs`` to be analyzed elsewhere.

- ``tracable_model.py``: This is where TracableModel, the model wrapper, is defined and the tracing logic mounted.

- ``pipeline.py``: the pipeline code.

- ``model_config.py``: A place to globally manage model configs. Useful since we may need to refrence the same parameters on more than one occasion.

- ``trace_nn.py``: Where the tracing logic is defined. The NeuronTrace and NetworkTrace classes live here.

- ``analysis.ipynb``:  This jupyter notebook is is where various analyses and visualizations can be conducted.

- ``helpers.py``: Helper functions that make certain routine activities easier.

## Roadmap

Currently the project is at the proof of concept stage. The minimum viable proof of concept is defined as the capability to scan a simple binary classification model to produce  visualizations about its most salient inner workings. 

- Proof of Concept _In Progress_ (but making very good progress)
- Draft of white paper 
- Add support for multi-classification ANNs **COMPLETED**
- Continue to research and improve on the analysis and visualizations _In Progress_
- Investigate ways to optimize storage and reduce general overhead _In Progress_
- Abstract the specification to support as many model types as possible **COMPLETED** (though more model support to be added on a rolling basis)
- Improve user-friendliness of API **COMPLETED**
- Add benchmarks to measure overhead (A/B testing)
- Improve robustness and error handling
- Simplified API for tracing **COMPLETED**
- Contintue to refine network schema and codebase organization
- Continue to add support for more exotic model configurations

## Developments 
Model explainability is an active field. Here I will document interesting developments in the field and comment on how they pretain to this project. 

- **Komolgorov-Arnold Networks**: Perhaps one of the more exciting developments I've come across recently, [KANs](https://github.com/mintisan/awesome-kan) are a fresh approach to model architecture overall. They differ from Multi-Layer Perceptrons with their big innovation being that they do away with matrices of weights and fixed activation functions on nodes altogether and instead use learnable activation functions on edges. The neurons, or nodes, in KANs simply sum together a linear combination of univariate continous functions that collectively represent continous multivariate functions thanks to the Komolgorov-Arnold Representation Theorem.  The result of this elegantly streamlined mathematics is a less messy computational environment and a cleaner architecture for tracking.

How do KANs affect this current project? I don't see why the methodology proposed in this codebase must conflict with KANs. How the states are traced would change, and the code would have to be rearranged to parallel the architecture differences but the value of tracing and plotting that data would remain the same. Since KANs are said to result in simplified computational graphs, if anything they should just complement my approach. However it'll be some time before I can figure out how to map KANs alongside more traditional MLPs. Indeed, I would argue that we would still need to do something like I am proposing regardless of how much cleaner we make the actual model computations. 

- **Neuron Tracing and Active Learning Environment (NeuroTrALE)** : [NeuroTrALE](https://www.ll.mit.edu/sites/default/files/other/doc/2023-02/TVO_Technology_Highlight_45_NeuroTrALE.pdf) is a new open source framework put out by MIT's Lincoln Laboratory that uses supercomputers and machine learning to map real (biological) brain networks. I suspect that deep down there is some kind of analogy to be found in what they are successfully doing to map biological NNs. So it may be worth it to explore their line of thinking for inspiration. 
