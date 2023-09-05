import pandas as pd
import os
import logging
from prefect import flow, task

@task(retries=3,retry_delay_seconds=2)
def read_and_process_data(parent_dir: str, filename: str, logger) -> pd.DataFrame:
    """
    This function will read and prepare text for training
    """
    logger.debug(f"Received parent directory for data is {parent_dir}")
    logger.debug(f"Received train file name is {filename}")
    try:
        df = pd.read_csv(
            os.path.join(parent_dir, filename),
            dtype={"title": str, "description": str, "tag": str},
            index_col=None,
        )
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
    df["combined"] = df["title"] + " " + df["description"]
    df["combined"] = df["combined"].apply(lambda text: text.lower())
    df["combined_with_SEP"] = df["title"] + " [SEP] " + df["description"]
    df["combined_with_SEP"] = df["combined_with_SEP"].apply(lambda text: text.lower())
    return df
