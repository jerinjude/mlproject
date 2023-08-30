import logging
import mlflow
#from mlflow import log_metric, log_param, log_params, log_artifacts
from sklearn.metrics import roc_auc_score
import torch
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from airflow.decorators import task

# logging.debug('This is a debug message')
# logging.info('This is an info message')
# logging.warning('This is a warning message')
# logging.error('This is an error message')
# logging.critical('This is a critical message')

@task
def model_trainer(
    mlflow_experiment_name,
    num_epochs,
    model,
    optimizer,
    criterion,
    train_dataloader,
    valid_dataloader,
    logger,
    lr,
    dropout,
    metric,
    filename,
    n_model_layer,
    num_classes,
    rundir
):
    mlflow.start_run()
    mlflow.log_param("model layers", n_model_layer)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("dropout", dropout)

    writer = SummaryWriter()
    # mlflow.set_experiment(mlflow_experiment_name)
    logger.info("Training started")
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        for batch_X, batch_y in train_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            print(outputs)
            loss = criterion(
                outputs, batch_y
            )  # Use class labels, not one-hot encoded targets
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        with torch.no_grad():
            model.eval()  # Set the model to evaluation mode
            valid_loss = 0.0
            for batch_X, batch_y in valid_dataloader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(valid_dataloader)
        avg_train_loss = train_loss / len(train_dataloader)

        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {avg_train_loss:.4f}"
        )
        logging.info(
            f"Epoch [{epoch+1}/{num_epochs}] - Validation Loss: {avg_valid_loss:.4f}"
        )

        # Log loss values for this epoch
        writer.add_scalar("Loss/train", avg_train_loss, epoch)
        writer.add_scalar("Loss/valid", avg_valid_loss, epoch)


        mlflow.log_metric("train_loss", avg_train_loss)
        mlflow.log_metric("valid_loss", avg_valid_loss)
        class_names = [
            "computer-vision",
            "graph-learning",
            "reinforcement-learning",
            "natural-language-processing",
            "mlops",
            "time-series",
        ]
    auc_sum=0
    for i, auc in enumerate(metric(valid_dataloader, model,num_classes)):
        auc_sum+=auc
        mlflow.log_metric(class_names[i] + "auc", auc)
    mlflow.log_metric("Average auc",(auc_sum/num_classes))
    mlflow.end_run()
    logging.info("Parameters and metric logged in mlfow")
    writer.flush()
    logger.info('Model training completed successfully')
    torch.save(model,rundir+"/model/" + filename)
