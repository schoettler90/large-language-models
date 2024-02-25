import os

from transformers import AutoModel


def load_and_save_model(model_name):
    """
    Loads a model from the Transformers library and saves it under a specified directory.

    Args:
      model_name: The name of the model to load.

    Returns:
      The loaded model.
    """

    # Load the model
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    # Create the output directory
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model
    model_path = os.path.join(model_dir, model_name)
    model.save_pretrained(model_path)

    return model


def load_model(model_name):
    """
    Loads a model from the Transformers library and saves it under a specified directory.

    Args:
      model_name: The name of the model to load.

    Returns:
      The loaded model.
    """

    # Load the model
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    return model


def main():
    model_name = "models/tiiuae/falcon-7b"
    model = load_model(model_name)
    print(model)

    print("Successfully loaded and saved model.")


if __name__ == "__main__":
    main()
