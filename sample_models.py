import torch
import torch.nn as nn
from model_config import config_iris

class IrisNN(nn.Module):
    """
        We're using the a model based on the classic iris dataset as a prototype.
        It haps the advantage of being a very well studied dataset with distinguished clusters.
    Its general features are:
    1.	Number of Instances:
	•	The dataset contains 150 instances (rows).
	2.	Number of Classes:
	•	There are three classes (species of iris flowers):
	•	Iris setosa
	•	Iris versicolor
	•	Iris virginica
	•	Each class has 50 instances.
	3.	Number of Features:
	•	The dataset has four features (attributes) for each instance:
	•	Sepal length (in centimeters)
	•	Sepal width (in centimeters)
	•	Petal length (in centimeters)
	•	Petal width (in centimeters)
	•	All features are continuous numeric values.
    """
    def __init__(self):
        super(IrisNN, self).__init__()
        input_size = config_iris.input_size
        hidden_size = config_iris.hidden_size
        output_size = config_iris.output_size
        layer_names = config_iris.layer_names

        setattr(self, layer_names['input'], nn.Linear(input_size, hidden_size))
        setattr(self, layer_names['hidden'], nn.Linear(hidden_size, hidden_size))
        setattr(self, layer_names['output'], nn.Linear(hidden_size, output_size))

        self.layer_names = layer_names

    def forward(self, x):
        x = torch.relu(getattr(self, self.layer_names['input'])(x))
        x = torch.relu(getattr(self, self.layer_names['hidden'])(x))
        x = torch.softmax(getattr(self, self.layer_names['output'])(x), dim=1)
        return x