import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataHandler:
    """
    A utility class for handling data preprocessing and loading for machine learning models.

    Attributes:
        dataset_type (str): The type of dataset to generate ('classification' or 'iris').
        n_samples (int): Number of samples to generate for the synthetic dataset.
        n_features (int): Number of features for each sample.
        n_classes (int): Number of classes in the target variable.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
        scaler (object): Scaler object for standardizing the features. Defaults to StandardScaler.
        X_train (torch.Tensor): Tensor containing the training data after scaling.
        X_test (torch.Tensor): Tensor containing the test data after scaling.
        y_train (torch.Tensor): Tensor containing the training labels.
        y_test (torch.Tensor): Tensor containing the test labels.
    """

    def __init__(self, dataset_type='classification', n_samples=1000, n_features=20, n_classes=2, 
                test_size=0.2, random_state=42, scaler=None, n_informative=None):
        """
        Initializes the DataHandler with configurable parameters.

        Args:
            dataset_type (str, optional): The type of dataset to generate ('classification' or 'iris'). Defaults to 'classification'.
            n_samples (int, optional): Number of samples to generate for the synthetic dataset. Defaults to 1000.
            n_features (int, optional): Number of features for each sample. Defaults to 20.
            n_classes (int, optional): Number of classes in the target variable. Defaults to 2.
            test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            scaler (object, optional): Scaler object for standardizing the features. If None, defaults to StandardScaler.
            n_informative (int, optional): Number of informative features to generate. Must satisfy n_classes * n_clusters_per_class <= 2**n_informative.
        """
        self.dataset_type = dataset_type
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = scaler if scaler else StandardScaler()  # Default to StandardScaler if none is provided
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_informative = n_informative if n_informative else min(n_features, max(2, n_classes))  # Safe default

    def generate_data(self):
        """
        Generates a synthetic dataset or loads a predefined dataset, performs train-test split,
        scales the features, and converts the data to PyTorch tensors.
        """
        if self.dataset_type == 'classification':
            X, y = make_classification(n_samples=self.n_samples, n_features=self.n_features, 
                                    n_classes=self.n_classes, random_state=self.random_state, 
                                    n_informative=self.n_informative)
        elif self.dataset_type == 'iris':
            iris = load_iris()
            X = iris.data
            y = iris.target
        else:
            raise ValueError(f"Unsupported dataset_type: {self.dataset_type}")

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        # Scale the features
        self.X_train = torch.tensor(self.scaler.fit_transform(X_train), dtype=torch.float32)
        self.X_test = torch.tensor(self.scaler.transform(X_test), dtype=torch.float32)
        
        # Convert labels to tensors
        self.y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        self.y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    def get_data_loaders(self, batch_size=16, shuffle_train=True):
        """
        Creates PyTorch DataLoader objects for the training and test datasets.

        Args:
            batch_size (int, optional): Number of samples per batch. Defaults to 16.
            shuffle_train (bool, optional): Whether to shuffle the training data. Defaults to True.

        Returns:
            tuple: A tuple containing the training DataLoader and the test DataLoader.
        """
        # Create TensorDatasets for training and test sets
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

# Instantiate DataHandler with dataset_type parameter
data_handler_iris = DataHandler(dataset_type='iris', n_samples=500, n_features=10, n_classes=3, n_informative=5, scaler=StandardScaler())
# Generate the Iris dataset
data_handler_iris.generate_data()
# Create the DataLoaders for the Iris dataset
train_loader_iris, test_loader_iris = data_handler_iris.get_data_loaders(batch_size=32)