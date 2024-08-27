# ANN_Tracing
![Image of a artificial neural network over graph paper](/assets/ann_graph_paper.jpg)


## Table of Contents
- [Overview](#overview)
- [Inspiration & Rationale](#inspiration--rationale)
- [API Reference](#api--reference)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Files](#files)
- [Roadmap](#roadmap)
- [Attributions](#attributions)
- [Collaborate](#collaborate)
- [Note on terminology](#terminology)
## Overview

If you are looking for more technical details about the API and how to navigate, use, and run the code, jump to the **API Reference, Usage, Dependencies, & Files** sections. For a more high level exposition on the project and its motivations and aspirations, read on.

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


## API Reference

There are really only three key components to this codebase. The ``TraceableNN``, ``NeuronTrace``, and ``NetworkTrace`` classes.

``TraceableNN`` is essentially a decorator for a regular pytoch model that injects the tracing logic onto a pytorch nn model. ``NeuronTrace`` traces each node in the network, and is a subclass of ``NetworkTrace`` which tracks the entire model. 

### TraceableNN 

``TraceableNN`` is essentially a wrapper for a model that takes hyperparameters and decorates it with recording logic. For all other purposes it behaves like a regular ANN you might define in pytorch. 

The implementation of TracableNN is planned to change, with the intent for it to eventually become more abstracted. The ultimate vision for it is that you pass a recipe for any model into it and the tracing logic gets injected and wrapped around any kind of model. 

Upon initialization, ``TraceableNN`` will perform a dry run to initialize the ``network_trace``, the basic framework we will use for future analysis. It assigns every neuron in the model a ``neuron_id`` which is a string of the form: ``n_0, n_1, ..., n_m`` where ``m`` is the cardinal of the set ``n``. Each neuron has a ``NeuronTrace`` object that can capture various statistical metrics about it.  

#### ``TraceableNN`` properties:

- ``input_size, hidden_size, output_size``: (Note: subject to change) These are just regular model config settings that you would provide to specify the dimensions of your model. They and similar configuration variables are globally managed in `model_config.py`. In future iterations I plan to include a more flexible API for feeding in model specs to support a diversity of different model configurations. 

- ``num_epochs``: The number of epochs for your training loop. Globally managed via `model_config.py`. 

- ``epoch_intervals``: Types: ``[Optional]`` ``int``, ``List[int]``, defaults to ``None``. Controls the frequency at which the recording/tracing logic is applied during model training. If a positive nonzero ``int`` is provided, recording will occur at every nth interval where n is the provided ``int``. If ``-1`` is provided, only the last epoch will be recorded/traced. If ``List[int]`` is provided, tracings will occur only at those specified epochs. If no value is provided, scans will occur during every epoch. **Note: Recording for all epochs is not recommended for large models or datasets**

- ``neuron_ids``: Tracks the ids corresponding to each neuron in the network, so they can later be mapped and associated to their respective layers. Upon initialization, ``TraceableNN`` will pass over the model and assign every neuron a unique identifier.

- ``neural_index``: The neural_index dictionary stores the start and end indices of neurons for each layer in the network. For example, if the input layer has 20 neurons, the neural_index for this layer might be (0, 19). This means that neurons in the input layer are indexed from 0 to 19. The neural_index property allows you to manage and organize the tracing data more effectively by ensuring that each neuron has a unique and consistent identifier across different layers and epochs. This is necessary to properly assign neuron_ids and maintain an accurate inventory of neurons across layers.

### NeuronTrace
``NeuronTrace`` is the tracing logic's concept of a particular instance of a neuron. More precisely, each neuron trace is a canonical form of the neuron, capturing regularized metrics of across the training lifecycle. 

#### ``NeuronTrace`` properties:

- ``signature``: A uniquely identifying string of the form ``[epoch_key][layer_key][neuron_key]``. Since the network trace is essentially a big hash lookup table, stamping each neuron across the training lifecycle with a signature like this makes it uniquely identifiable both in "time" and "space" and provides a slick means to access any neuron across the model lifecycle. Suppose I want to look up the 20th neuron in the hidden layer at epoch 50, I would just have to write ``my_neuron = network_trace[E_50][L_hidden_1][n_20]``

- ``input_neurons``: **DEPRECATED**. Tracks the "canonical connections" of the neuron to its immediate input neurons or features. This approach has been phased out after I realized a more complicated analysis of the neuron's weights is necessary to determine "input neurons of most significance".

- ``bread_crumbs``: WIP. Currently not in use. But the idea of breadcrumbs is similar to input_neurons. But eventually these will be calculated from an analysis of the most consistent strengths of connections, leading to a picture of what neurons contributed most to a classification result. I am now realizing that these breadcrumbs do not belong on the NeuronTrace class, however. They should be considered global properties.

- ``activation, weight,`` and ``bias`` +  ``_metrics``: Captures a basic statistical profile of the neuron. Currently these metrics are offline and unused while I figure out what to do with them. More metrics can be added arbitrarily. These will probably be rearranged in future iterations. 


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

#### ``NetworkTrace`` properties: 


- ``drop_batches``: Types: ``[Optional]`` ``bool`` defaults to ``True``. If set to ``True`` intermediate metadata will be dropped after computing final metrics. This is to help reduce storage footprint. Otherwise, intermediate metadata will be conserved across the training lifecycle and exported along with the network_trace. Similar drop flags are likely to be added for other categories of data e.g ``drop_weights``, ``drop_biases``, etc, if for some reason the user wants to discard these (potentially heavy) collections after computing over them.

- ``history``: WIP. The history attribute tracks transient, nonlearnable parameters, which for now are essentially just the activation values. Every neuron is indexed by signature, and its activations are updated and stored by ``batch_id``. Basic statistics would then be calculated from this history, placed on the NeuralTrace objects of the corresponding signature, and the history dictionary reset to empty. I realized later however that storing activation values during training is not very useful, a space hog, and quite cumbersome. The plan for history in the future is to measure it during inference mode on the now pre-trained model. Details of this plan are forthcoming.

- ``trace``: This is the actual skeleton of the schema that will hold with all the data on it. It has the form:

``
Epoch: {
    Layer: {
        ...,
        Neuron:{
            ...,
        }
        ...,
    },
    ...,
}
``
Various properties that are appropriate to the given level can be placed on it. So for example, the weights are recorded on the Neuron level, for every neuron, at every epoch, etc. 

I'm still playing around with what kind of data should go where, so expect this layout to change.

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


## Modifying the ANN model
While the ideal goal of this project is to have an abstract specification that works with any model, the reality at the moment is that the code in its current state is probably closely coupled to the exact parameters of the ``TracableNN`` class and will break if exposed to anything too fundamentally different in layout. Once I get a good proof of concept, hopefully I will figure out how to make this work for all types of models. 



## Dependencies

For a list of required modules, see:
``
python requirements.txt 
``
A few libraries not listed in the requirements.txt
- ``pandas                    2.2.2 ``
- ``adjusttext                1.2.0 ``

## Files

- ``data.py``: This is where you would set up the data for the model. I am currently using a synthetic binary classification data set using sci-kit learn's make_classification function. Originally I thought it really didn't matter what sort of data was being used, only that it was simple at first. Now I realize I probably should have chosen a more semantic data set as this could have made it easier to identify meaningful states. I am planning a major overhaul to this part of the code however .

- ``main.py``: the main process. Everything comes together here. The model is initialized and declared, the data imported and provided to it, and the training loop executed. Results are then saved away to ``/outputs`` to be analyzed elsewhere.

- ``model.py``: This is where TracableNN, the model wrapper, is defined and the tracing logic mounted.

- ``model_config.py``: A place to globally manage model configuration. Useful since we may need to refrence the same parameters on more than one occasion.

- ``trace_nn.py``: Where the tracing logic is defined. The NeuronTrace and NetworkTrace classes live here.

- ``analysis.ipynb``: Where I imagine most users will spend most of their time once I create a better pipeline for integrating and mounting any given model for tracing. This jupyter notebook will cast the network trace into a dataframe and is where various analyses can be conducted.

- ``plot.py`` Various functions related to plotting visualizations. Currently not seeing much use as I revisit the whole idea about what approach is best for capturing meaningful information about model metadata.

- ``helpers.py``: Helper functions that make certain routine activities easier. Currently there are functions here that make navigating the string to integer casting and vice versa that is necessary to do hash lookups efficiently.

- ``inspection.py``: Various debugging and observation tools for printing out data when needed.

## Roadmap

Currently the project is at the proof of concept stage. The minimum viable proof of concept is defined as the capability to scan a simple binary classification model to produce  visualizations about its most salient inner workings. 

- Proof of Concept (In Progress)
- Draft of white paper 
- Add support for multi-classification ANNs
- Continue to research and improve on the analysis and visualizations (In Progress)
- Investigate ways to optimize storage and reduce general overhead (In Progress)
- Generalize into an implementation-agnostic specification that supports as many model types as possible 
- Improve usage (either by providing better scripting logic so sourcefiles don't need to be manually modified and can be set via terminal commands, or by providing a GUI )
- Add benchmarks to measure overhead (A/B testing)

## Attributions:
When the idea for this project first occured to me, I bounced it off ChatGPT as I have come to find using it as a sounding board  is often a helpful way to flesh thoughts out. My initial prompt was very high level and abstract and I did not ask for code. But to my surprise it began to generate code for it. At that point I was almost as interested to see if the AI could generate working code for my idea as I was in the idea itself. 

While ChatGPT has a share conversation feature, I was getting this error "Sharing conversations with images or audio is not yet supported", and so to share the whole chat I would have to go through a lengthier process of exporting my entire chat history, sorting through a big JSON object of the conversations, finding that session and making it presentable with all its markdown and structure. 

Since this involves having to download my my entire conversational history from OpenAI and working through it it's not a competely trivial task. However I do plan on sharing the convo at some point. Both for the sake of academic transparency but also because I think it is an interesting document. If you are interested in this project and AI, I recommend  browsing this chat history when I do get it up as it is a good demonstration of how AI and software developers can fruitfully collaborate. The developer specifies the overall structure and plan, makes high level organizational decisions, provides feedback and critique, verbally specifies detailed requirements, and makes certain creative or targeted code changes, and meanwhile the AI does a lot of the heavy lifting, initial drafting, prototyping, and boilerplating, while also occasionally providing genuinely valuable feedback and conceptual recommendations. 

When you look at this chat, it becomes clear when the AI reaches its limit and by comparing the code it generates to this codebase you can see where the two depart. Without using AI generated code this project would have probably taken me 3-6 months to do completely manually. It's amazing how quickly it allowed me to bootstrap my ideas, try out many different approaches at reduced effort and cost, and vastly accelerate knowledge production.

## Collaborate: 
If you would like to contribute to this project please reach out! I could especially use help from people with a strong background in data analysis and linear algebra. I haven't yet 100% realized my vision for the project, and before I open the doors up to open sourcing I would like to at least refactor a few areas to make it less coupled on one model. Hopefully this time will come soon though!

## Terminology: 
The term "staining" came to mind when I first thought of this idea. It is meant to invoke the tissue staining methods used in histology by biologists studying cells. They add certain biochemical markers to tissues to detect various cellular differences. Since this project tries to isolate and color code different structures in artificial neural networks, the metaphor felt apt. I've grown less attached to this imagery over time, but you may still see references to a "staining algorithim" here and there.
