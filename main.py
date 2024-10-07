from pipeline import TraceablePipeline
from persistence_handler import PersistenceHandler
from model_config import config_iris
from sample_models import IrisNN

def main():
    pipeline_path = 'outputs/trained_pipeline.pkl'
    model_path = 'outputs/trained_model.pth'

    try:
        # Load the saved pipeline
        traceable_pipeline = PersistenceHandler.load_pipeline(pipeline_path)
    except FileNotFoundError:
        print(f"No saved pipeline found at {pipeline_path}. Initializing and training a new pipeline.")
        # Initialize and train a new pipeline
        traceable_pipeline = TraceablePipeline(
            model=IrisNN(),
            config=config_iris,
            model_type='multi_class_classification',
            dataset_type='iris'
        )
        traceable_pipeline.train()
        PersistenceHandler.save_pipeline(traceable_pipeline, pipeline_path)
        PersistenceHandler.save_model(traceable_pipeline.model, model_path )
    # Evaluate the model using the loaded or trained pipeline
    traceable_pipeline.evaluate()

if __name__ == "__main__":
    main()