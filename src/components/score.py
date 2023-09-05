from sklearn.metrics import roc_auc_score
from prefect import flow, task

@flow
def multiclass_auc(data_loader, model,num_classes):
    true_labels = []
    predicted_probs = []

    # Set the model to evaluation mode
    model.eval()

    # Iterate through the DataLoader
    for inputs, labels in data_loader:
        # Forward pass to get predicted probabilities
        outputs = model(inputs)
        # print('outputs ',outputs.detach().numpy())
        # print('labels',labels.numpy())
        # Convert tensor to NumPy array and append to the list
        true_labels.extend(labels.numpy())
        predicted_probs.extend(outputs.detach().numpy())
    # Calculate AUC for each class
    auc_scores = list()
    for class_index in range(
        num_classes
    ):  # Replace 'num_classes' with the actual number of classes
        true_class_labels = [
            1 if label[class_index] == 1 else 0 for label in true_labels
        ]
        # print(true_class_labels)
        # true_class_labels = true_class_labels.tolist()
        class_probs = [prob[class_index] for prob in predicted_probs]
        # print(class_probs)
        auc = roc_auc_score(true_class_labels, class_probs)
        auc_scores.append(auc)
    return auc_scores
