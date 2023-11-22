import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch

from ThreeHeadClassifier import ThreeHeadClassifier
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


def fit(
        model,
        train_embeddings,
        train_measurement_label,
        train_location_label,
        train_description_label,
        loss_fn,
        optimizer,
        num_epochs=30,
        batch_size=64,
):
    # Calculate the number of batches
    num_batches = train_embeddings.shape[0] // batch_size

    for epoch_ind in range(num_epochs):
        epoch = epoch_ind + 1
        # Set the model to training mode
        model.train()
        # Shuffle the data
        shuffled_indices = torch.randperm(train_embeddings.shape[0])
        # Get the shuffled embeddings and labels
        shuffled_embeddings = train_embeddings[shuffled_indices]
        shuffled_measurement_label = train_measurement_label[shuffled_indices]
        shuffled_location_label = train_location_label[shuffled_indices]
        shuffled_description_label = train_description_label[shuffled_indices]
        # Loop through the batches

        for i in range(num_batches):
            # Get the batch embeddings and labels
            batch_embeddings = shuffled_embeddings[i * batch_size:(i + 1) * batch_size]
            batch_measurement_label = shuffled_measurement_label[i * batch_size:(i + 1) * batch_size]
            batch_location_label = shuffled_location_label[i * batch_size:(i + 1) * batch_size]
            batch_description_label = shuffled_description_label[i * batch_size:(i + 1) * batch_size]

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

            # Print the loss every 5 epochs
            if epoch == 1 and i == 0:
                print(f"Epoch: {epoch}, Loss 1: {loss1: .2f}, Loss 2: {loss2: .2f}, Loss 3: {loss3: .2f},"
                      f" Total loss: {loss}")

            if epoch % 10 == 0 and i == len(range(num_batches)) - 1:
                print(f"Epoch: {epoch}, Loss 1: {loss1: .2f}, Loss 2: {loss2: .2f}, Loss 3: {loss3: .2f},"
                      f" Total loss: {loss: .2f}")

    return model


def main():
    import config

    print("Device: ", DEVICE)

    data_path = config.EMBEDDINGS_DATA_PATH
    train_df, test_df = load_and_split_data(data_path)

    get_number_of_classes = (lambda x: len(x.unique()))
    num_classes_measurement = get_number_of_classes(train_df['measurement'])
    num_classes_location = get_number_of_classes(train_df['location'])
    num_classes_description = get_number_of_classes(train_df['description'])

    train_embeddings = torch.tensor(train_df['embeddings'].tolist())
    # test_embeddings = torch.tensor(test_df['embeddings'].tolist())

    train_measurement_label, test_measurement_label = one_hot_encode(train_df['measurement'], test_df['measurement'])
    train_location_label, test_location_label = one_hot_encode(train_df['location'], test_df['location'])
    train_description_label, test_description_label = one_hot_encode(train_df['description'], test_df['description'])

    input_dim = train_embeddings.shape[1]

    # Initialize the model
    model = ThreeHeadClassifier(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT,
        num_classes_head1=num_classes_measurement,
        num_classes_head2=num_classes_location,
        num_classes_head3=num_classes_description,
    ).to(DEVICE)

    print(model)

    # Define the loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model = fit(
        model,
        train_embeddings,
        train_measurement_label,
        train_location_label,
        train_description_label,
        loss_fn,
        optimizer,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # Save the model
    model_save_path = config.MODEL_SAVE_PATH
    torch.save(model.state_dict(), model_save_path)

    print("Model saved to: ", model_save_path)


if __name__ == '__main__':
    main()
