from model_config import config_iris
from sample_models import IrisNN
from pipeline import TraceablePipeline

def main():
    # Initialize the pipeline
    tracable_pipeline = TraceablePipeline(
        model=IrisNN(),
        config=config_iris,
        model_type='multi_class_classification'
    )

    # Train the model
    tracable_pipeline.train()

    # Evaluate the model
    tracable_pipeline.evaluate()

    # Save the results
    tracable_pipeline.save_results()

if __name__ == "__main__":
    main()