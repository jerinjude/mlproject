from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
import torch
import os
import pandas as pd
import ast
#from prefect import flow, task

#@task
def featurization(text, model):
    embeddings = model.encode(text,show_progress_bar=False)
    return embeddings

#@task
def create_dataloader(
    run_dir,
    df_train,
    df_valid,
    SEP,
    batch_size,
    logger,
    model_name="sentence-transformers/all-mpnet-base-v2",
):
    train_path = f"{run_dir}/temp_files/train.csv"
    valid_path = f"{run_dir}/temp_files/valid.csv"
    file_exists = True
    for path in [train_path, valid_path]:
        filename = path.split("/")[-1]
        if os.path.exists(f"{run_dir}/temp_files/train.csv"):
            logger.info(f"The file {filename} exists.")
        else:
            file_exists = False
            logger.info(f"The file {filename} does not exist.")

    if not file_exists:
        logger.info(f"Creating new csv files with processed data")
        label_encoder = LabelEncoder()
        label_encoder.fit(df_train["tag"])
        model = SentenceTransformer(model_name)
        for df in [df_train, df_valid]:
            if SEP:
                df["bert_features_combined"] = df["combined_with_SEP"].apply(
                    lambda x: featurization(x, model)
                )
            else:
                df["bert_features_combined"] = df["combined"].apply(
                    lambda x: featurization(x, model)
                )
            df['bert_features_combined'] = df['bert_features_combined'].apply(lambda x: ' '.join(map(str, x)))
            df["label_int"] = label_encoder.transform(df["tag"])
        try:
            df_train.to_csv(f"{run_dir}/temp_files/train.csv",index=False)
            df_valid.to_csv(f"{run_dir}/temp_files/valid.csv",index=False)
        except Exception as e:
            logger.error(f"An error occurred while saving csv: {str(e)}", exc_info=True)
    else:
        try:
            df_train = pd.read_csv(f"{run_dir}/temp_files/train.csv",index_col=False,dtype={'bert_features_combined':str})
            df_valid = pd.read_csv(f"{run_dir}/temp_files/valid.csv",index_col=False,dtype={'bert_features_combined':str})
            logger.info('Loaded existing csv files')
            #logger.debug(f'df_train info str({df_train.info()})')
            #logger.debug(f'df_valid info str({df_valid.info()})')
        except Exception as e:
            logger.error(
                f"An error occurred while reading csv: {str(e)}", exc_info=True
            )
    df_train['bert_features_combined'] = df_train['bert_features_combined'].apply(lambda x: list(map(float, x.split())))
    df_valid['bert_features_combined'] = df_valid['bert_features_combined'].apply(lambda x: list(map(float, x.split())))
    onehot_encoder = OneHotEncoder(handle_unknown="ignore")
    label_int_train = df_train["label_int"].values.reshape(-1, 1)
    label_int_valid = df_valid["label_int"].values.reshape(-1, 1)
    onehot_encoder.fit(label_int_train)

    y_train = torch.tensor(onehot_encoder.transform(label_int_train).toarray())
    y_valid = torch.tensor(onehot_encoder.transform(label_int_valid).toarray())

    X_train = torch.tensor(df_train["bert_features_combined"])
    X_valid = torch.tensor(df_valid["bert_features_combined"])

    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(X_valid, y_valid)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, valid_dataloader

#@task
def text_processing_pred(title,desc,bert_model):
    text = title + " " + desc
    text = featurization(text, bert_model)
    text = torch.tensor(text)
    return text


