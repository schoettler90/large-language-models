import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch

from MultiHeadClassifier import MultiHeadClassifier
from LabelEncoder import LabelEncoder
from preprocessing import get_target
import config

# set the device to cuda if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = config.HIDDEN_DIM
LEARNING_RATE = config.LEARNING_RATE
NUM_EPOCHS = config.NUM_EPOCHS
BATCH_SIZE = config.BATCH_SIZE
DROPOUT = config.DROPOUT
WEIGHT_DECAY = config.WEIGHT_DECAY


def load_and_split_data(data_path, test_size=0.2, random_state=42):
    # Read in the data
    df = pd.read_pickle(data_path)

    # Split the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)

    return train_df, test_df


def one_hot_encode(train_series, test_series):
    # One-hot encode the labels
    enc = OneHotEncoder(handle_unknown='ignore')
    train_one_hot = enc.fit_transform(train_series.to_frame()).toarray()
    test_one_hot = enc.transform(test_series.to_frame()).toarray()

    # convert it to pytorch tensor
    train_labels = torch.tensor(train_one_hot)
    test_labels = torch.tensor(test_one_hot)

    return train_labels, test_labels


def fit_one_hot_encoder(train_series):
    # One-hot encode the labels

    encoder = LabelEncoder()
    encoder.fit(train_series)

    return encoder


def return_train_print(epoch, loss1, loss2, loss3, loss) -> str:
    output = f"Epoch: {epoch}, Losses:  [{loss1: .2f}, {loss2: .2f}, {loss3: .2f}], Total Loss: {loss: .2f}. "
    return output


def return_eval_print(epoch, accuracy1, accuracy2, accuracy3, total_accuracy) -> str:
    output = (f"Epoch: {epoch}, Eval Accuracies: [{accuracy1: .2f}, {accuracy2: .2f}, {accuracy3: .2f}] "
              f"Total eval accuracy: {total_accuracy: .2f}")
    return output


def softmax_to_one_hot(softmax_probs):
    _, max_indices = torch.max(softmax_probs, 1)
    one_hot = torch.zeros_like(softmax_probs)
    one_hot.scatter_(1, max_indices.view(-1, 1), 1)
    return one_hot


def get_accuracy(predictions: torch.Tensor, labels: torch.Tensor):
    """
    :param predictions: torch.Tensor of shape (num_examples, num_classes)
    :param labels: torch.Tensor of shape (num_examples, num_classes)
    :return:
        float: accuracy across all examples

    """
    # argmax the predictions
    _, predictions = torch.max(predictions, 1)
    # argmax the labels
    _, labels = torch.max(labels, 1)

    # calculate the accuracy
    accuracy = (predictions == labels).sum().item() / len(labels)

    return accuracy


def fit_multi_head_model(
        model: torch.nn.Module,
        train_embeddings: torch.Tensor,
        train_labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        eval_embeddings: torch.Tensor,
        eval_labels: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 100,
        batch_size: int = 16,
):
    # Calculate the number of batches
    num_batches = train_embeddings.shape[0] // batch_size

    # unpack the labels
    train_measurement_label, train_location_label, train_description_label = train_labels
    eval_measurement_label, eval_location_label, eval_description_label = eval_labels

    # get initial accuracy for evaluation set
    model.eval()

    # Get the model predictions
    output1, output2, output3 = model(eval_embeddings.to(DEVICE))

    # calculate the accuracy
    measurement_accuracy = get_accuracy(output1, eval_measurement_label.to(DEVICE))
    location_accuracy = get_accuracy(output2, eval_location_label.to(DEVICE))
    description_accuracy = get_accuracy(output3, eval_description_label.to(DEVICE))

    # calculate total accuracy, when all 3 heads are correct
    total_accuracy = measurement_accuracy * location_accuracy * description_accuracy

    eval_print = return_eval_print(
        0,
        measurement_accuracy,
        location_accuracy,
        description_accuracy,
        total_accuracy,
    )

    print(eval_print)

    # Loop through the epochs
    for epoch_ind in range(num_epochs):
        epoch = epoch_ind + 1

        # Set the model to training mode
        model.train()

        # Shuffle the data
        shuffled_indices = torch.randperm(train_embeddings.shape[0])

        # Get the shuffled embeddings
        shuffled_embeddings = train_embeddings[shuffled_indices]

        # Get the shuffled labels
        shuffled_measurement_label = train_measurement_label[shuffled_indices]
        shuffled_location_label = train_location_label[shuffled_indices]
        shuffled_description_label = train_description_label[shuffled_indices]

        # initialize the losses to high number
        loss1 = 1000
        loss2 = 1000
        loss3 = 1000
        loss = loss1 + loss2 + loss3

        # Loop through the batches
        for i in range(num_batches):
            # Get the batch embeddings and labels
            batch_embeddings = shuffled_embeddings[i * batch_size: (i + 1) * batch_size]
            batch_measurement_label = shuffled_measurement_label[i * batch_size: (i + 1) * batch_size]
            batch_location_label = shuffled_location_label[i * batch_size: (i + 1) * batch_size]
            batch_description_label = shuffled_description_label[i * batch_size: (i + 1) * batch_size]

            # Zero out the gradients
            optimizer.zero_grad()
            # Get the model predictions
            output1, output2, output3 = model(batch_embeddings.to(DEVICE))

            # Calculate the loss for each head
            loss1 = loss_fn(output1, batch_measurement_label.to(DEVICE))
            loss2 = loss_fn(output2, batch_location_label.to(DEVICE))
            loss3 = loss_fn(output3, batch_description_label.to(DEVICE))

            # Calculate the total loss
            loss = loss1 + loss2 + loss3

            # Backpropagate the loss
            loss.backward()
            # Update the parameters
            optimizer.step()

        if epoch % 10 == 0:
            # Evaluate the model
            train_print = return_train_print(epoch, loss1, loss2, loss3, loss)

            model.eval()
            # Get the model predictions

            output1, output2, output3 = model(eval_embeddings.to(DEVICE))
            # argmax the predictions and convert them to numpy array

            # convert evaluation labels to numpy array

            # calculate the accuracy
            measurement_accuracy = get_accuracy(output1, eval_measurement_label.to(DEVICE))
            location_accuracy = get_accuracy(output2, eval_location_label.to(DEVICE))
            description_accuracy = get_accuracy(output3, eval_description_label.to(DEVICE))

            # calculate total accuracy, when all 3 heads are correct
            total_accuracy = measurement_accuracy * location_accuracy * description_accuracy

            eval_print = return_eval_print(
                epoch,
                measurement_accuracy,
                location_accuracy,
                description_accuracy,
                total_accuracy,
            )

            print(train_print + eval_print)

    return model


def main():
    import config

    print("Device: ", DEVICE)

    # Load the data
    data_path = config.EMBEDDINGS_DATA_PATH
    train_df, test_df = load_and_split_data(data_path)

    # get the number of classes for each head
    get_number_of_classes = (lambda x: len(x.unique()))
    num_classes_measurement = get_number_of_classes(train_df['measurement'])
    num_classes_location = get_number_of_classes(train_df['location'])
    num_classes_description = get_number_of_classes(train_df['description'])

    # convert the embeddings to tensors
    train_embeddings = torch.tensor(train_df['embeddings'].tolist())
    test_embeddings = torch.tensor(test_df['embeddings'].tolist())

    # fit the encoders
    measurement_encoder = fit_one_hot_encoder(train_df['measurement'])
    location_encoder = fit_one_hot_encoder(train_df['location'])
    description_encoder = fit_one_hot_encoder(train_df['description'])

    # one-hot encode the labels
    train_measurement_hot = torch.tensor(measurement_encoder.transform(train_df['measurement']))
    test_measurement_hot = torch.tensor(measurement_encoder.transform(test_df['measurement']))

    train_location_hot = torch.tensor(location_encoder.transform(train_df['location']))
    test_location_hot = torch.tensor(location_encoder.transform(test_df['location']))

    train_description_hot = torch.tensor(description_encoder.transform(train_df['description']))
    test_description_hot = torch.tensor(description_encoder.transform(test_df['description']))

    # get the input dimension for the model
    input_dim = train_embeddings.shape[1]

    model_params = {
        'input_dim': input_dim,
        'hidden_dim': HIDDEN_DIM,
        'heads_dims': (num_classes_measurement, num_classes_location, num_classes_description),
        'dropout': DROPOUT,
    }

    # Initialize the model
    model = MultiHeadClassifier(
        **model_params,
    ).to(DEVICE)

    print(model)

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model = fit_multi_head_model(
        model=model,
        train_embeddings=train_embeddings,
        train_labels=(train_measurement_hot, train_location_hot, train_description_hot),
        eval_embeddings=test_embeddings,
        eval_labels=(test_measurement_hot, test_location_hot, test_description_hot),
        loss_fn=loss_fn,
        optimizer=optimizer,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # Save the model
    model_save_path = config.MODEL_SAVE_PATH
    torch.save(model.state_dict(), model_save_path)
    # save the model parameters
    with open('models/model_params.json', 'w') as f:
        json.dump(model_params, f)

    print("Model saved to: ", model_save_path)

    # evaluate the model
    eval_df = evaluate(
        model=model,
        test_df=test_df,
        encoders=(measurement_encoder, location_encoder, description_encoder)
    )

    # store the evaluation results
    eval_df.to_csv('data/evaluation_results.csv')


def evaluate(
        model: torch.nn.Module,
        test_df: pd.DataFrame,
        encoders: tuple[LabelEncoder, LabelEncoder, LabelEncoder]
):
    # convert the embeddings to tensors
    test_embeddings = torch.tensor(test_df['embeddings'].tolist())
    test_targets = test_df['target'].tolist()

    # get predictions
    model.eval()
    output1, output2, output3 = model(test_embeddings.to(DEVICE))

    # convert the predictions to one-hot
    measurement_hot = softmax_to_one_hot(output1).cpu().numpy()
    location_hot = softmax_to_one_hot(output2).cpu().numpy()
    description_hot = softmax_to_one_hot(output3).cpu().numpy()

    # inverse transform the one-hot to labels
    measurement_labels_pred = encoders[0].inverse_transform(measurement_hot, return_type='pandas')
    location_labels_pred = encoders[1].inverse_transform(location_hot, return_type='pandas')
    description_labels_pred = encoders[2].inverse_transform(description_hot, return_type='pandas')

    eval_df = pd.DataFrame({
        'measurement_pred': measurement_labels_pred,
        'location_pred': location_labels_pred,
        'description_pred': description_labels_pred,
    })

    eval_df['pred'] = eval_df.apply(
        lambda x: get_target(x['measurement_pred'], x['location_pred'], x['description_pred']), axis=1,
    )

    # get the actual labels
    eval_df['true'] = test_targets

    # calculate the accuracy
    accuracy = (eval_df['pred'] == eval_df['true']).sum() / len(eval_df)

    print(f"Validation Accuracy: {accuracy: .2f}")

    return eval_df


if __name__ == '__main__':
    main()
