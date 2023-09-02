import os
from datetime import datetime
import logging
from pathlib import Path
import sys
import mlflow
import torch.optim as optim
import torch.nn as nn
import transformers
transformers.logging.set_verbosity_error()
from prefect import flow, task


current_script_directory = Path(__file__).resolve().parent
print(current_script_directory)
parent_directory = Path(current_script_directory.parent / "components/")
print(parent_directory)
sys.path.append(parent_directory)
sys.path.append("/home/vboxuser/mlprojects/sample/src/components")
sys.path.append("/home/vboxuser/mlprojects/sample/src")


from data_ingestion import read_and_process_data
from data_transformation import create_dataloader
from model_arch import *
from model_trainer import model_trainer
from score import multiclass_auc
import logger

@task
def create_dirs():
    run_dir = f"./run_dir"
    model_dir = f"{run_dir}/model"
    log_dir = f"{run_dir}/logs"
    intermediate_file_dir = f"{run_dir}/temp_files"

    directories = {
        "rundir": run_dir,
        "model directory": model_dir,
        "log directory": log_dir,
        "intermediate_file_dir": intermediate_file_dir,
    }
    for name, dir in directories.items():
        # Create the directory
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
                logging.info(f"{name} created successfully at '{dir}'")
            else:
                logging.warning(f"{name} at '{dir}' already exists.")
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}", exc_info=True)
@flow
def main(
    data_parent_dir,
    train_filename,
    valid_filename,
    input_size,
    hidden_size1,
    hidden_size2,
    num_classes,
    dropout_prob,
    learning_rate,
    num_epochs,
    mlflow_experiment_name,
    has_SEP_token,
    batch_size,
    n_model_layer,
):
    # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"./run_dir"
    model_dir = f"{run_dir}/model"
    log_dir = f"{run_dir}/logs"
    intermediate_file_dir = f"{run_dir}/temp_files"

    directories = {
        "rundir": run_dir,
        "model directory": model_dir,
        "log directory": log_dir,
        "intermediate_file_dir": intermediate_file_dir,
    }
    for name, dir in directories.items():
        # Create the directory
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
                logging.info(f"{name} created successfully at '{dir}'")
            else:
                logging.warning(f"{name} at '{dir}' already exists.")
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}", exc_info=True)


    df_train = read_and_process_data(data_parent_dir, train_filename, logging)
    df_valid = read_and_process_data(data_parent_dir, valid_filename, logging)

    train_dataloader, valid_dataloader = create_dataloader(
        run_dir,
        df_train,
        df_valid,
        has_SEP_token,
        batch_size,
        logging,
        model_name="sentence-transformers/all-mpnet-base-v2",
    )

    if n_model_layer == 2:
        model = TwoLayerNN(input_size, hidden_size1, num_classes, dropout_prob)
    elif n_model_layer == 3:
        model = ThreeLayerNN(
            input_size, hidden_size1, hidden_size2, num_classes, dropout_prob
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    metric=multiclass_auc
    model_name='model1'

    

    model_trainer(
        mlflow_experiment_name,
        num_epochs,
        model,
        optimizer,
        criterion,
        train_dataloader,
        valid_dataloader,
        logging,
        learning_rate,
        dropout_prob,
        metric,
        model_name,
        n_model_layer,
        num_classes,
        run_dir
    )


if __name__ == "__main__":
    main(
        "/home/vboxuser/mlprojects/data/made_with_ml",
        "dataset.csv",
        "holdout.csv",
        768,
        128,
        512,
        6,
        0.3,
        0.001,
        20,
        "ml_topic_classification_exp",
        True,
        32,
        3,
    )