import torch
import pickle

class PersistenceHandler:
    '''
        Handles persistence for model and pipeline. 
        Its responsibility is to save and load the pipeline or model.
        Note: While save/load_pipeline also saves/loads the model, save/load_model is also provided for granular use cases.
    '''
    @staticmethod
    def save_pipeline(pipeline, pipeline_path='outputs/trained_pipeline.pkl'):
        """Save the entire pipeline, including model, config, and network trace."""
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        print(f'Pipeline saved to {pipeline_path}')

    @staticmethod
    def load_pipeline(pipeline_path='outputs/trained_pipeline.pkl'):
        """Load the entire pipeline, restoring the model, config, and network trace."""
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        print(f'Pipeline loaded from {pipeline_path}')
        return pipeline

    @staticmethod
    def save_model(model, model_path='outputs/trained_model.pth'):
        """Save only the model's state dictionary."""
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')

    @staticmethod
    def load_model(model, model_path='outputs/trained_model.pth'):
        """Load only the model's state dictionary."""
        model.load_state_dict(torch.load(model_path))
        print(f'Model loaded from {model_path}')