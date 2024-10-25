import logging
import boto3
from botocore.client import Config
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import ChatOllama
import llm_prompt_evaluation.config as config
import pandas as pd
import csv


def setup_logger(name, level=logging.INFO):
    """
    This function returns the logger.

    :param name: name of the logger
    :param level: logging level
    :return: logger
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_prompt_input():
    """
    This function returns the prompt input for the model, can be used a RAG To get information about the input.

    :return: dictionary includes the prompt input
    """
    return {
        "databaseName": "bm-db-prototype",
        "tableName": "input",
        "columns": {
            "id": {
                "type": "bigint",
                "description": "ID of transaction"
            },
            "Tipologia": {
                "type": "string",
                "description": "type of transaction",
                "enums": ["Bolli", "Noleggio", "Lavaggio PickUp"]
            },
            "Servizio": {
                "type": "string",
                "description": "service related to transaction"
            },
            "Modo": {
                "type": "string",
                "description": "method of use related to transaction"
            },
            "Fornitore": {
                "type": "string",
                "description": "provider related to transaction"
            },
            "Data": {
                "type": "date",
                "description": "transaction creation date",
                "rules": ["max-interval: 3 months"]
            },
            "Importo pieno": {
                "type": "double",
                "description": "amout of transaction"
            }
        }
    }


def load_dataset_from_csv(upload_path, delimiter=";", quoting=csv.QUOTE_NONE, encoding='utf-8'):
    """
    This function loads the dataset from the CSV file.

    :param upload_path: path of the uploaded file
    :return: DataFrame
    """
    # Carica il CSV in un DataFrame
    df = pd.read_csv(upload_path, delimiter=delimiter, quoting=quoting, encoding=encoding, index_col=None)
    df["Risposta SQL Attesa"] = df["Richiesta Utente"]
    df["Richiesta Utente"] = df.index
    df.reset_index(drop=True, inplace=True)
    # df = df.iloc[:3]
    # df.reset_index(drop=True, inplace=True)

    return df


def routing_model(model_id, model_type="BEDROCK", metric_callbacks=None):
    """
    This function returns the model to be used for the prompt.

    :param model_id: model id
    :param model_type: model type Bedrock | Ollama
    :param metric_callbacks: metric callbacks
    :return: model, embedding model
    """

    if metric_callbacks is None:
        metric_callbacks = []

    if model_type.upper() != "BEDROCK":
        # inizializza modello Ollama
        model = ChatOllama(base_url=config.URL_MODEL,
                           model=model_id,
                           num_gpu=config.NUM_GPU,
                           num_thread=config.NUM_THREAD,
                           temperature=config.TEMPERATURE,
                           callbacks=metric_callbacks)

        # Inizializza il modello Ollama embeddings
        embeddings_model = OllamaEmbeddings(base_url=config.URL_MODEL,
                                            model=model_id,
                                            num_gpu=config.NUM_GPU,
                                            num_thread=config.NUM_THREAD,
                                            temperature=config.TEMPERATURE)
    else:
        # Inizializza il client Bedrock
        bedrock_config = Config(connect_timeout=config.AWS_CLIENT_CONNECT_TIMEOUT,
                                read_timeout=config.AWS_CLIENT_READ_TIMEOUT,
                                retries={'max_attempts': config.AWS_MAX_ATTEMPTS})
        bedrock_client = boto3.client('bedrock-runtime', region_name=config.AWS_REGION, config=bedrock_config)

        # Inizializza il modello LLM
        model = ChatBedrock(model_id=model_id,
                            client=bedrock_client,
                            model_kwargs={"temperature": config.TEMPERATURE},
                            callbacks=metric_callbacks
                            )
        # Inizializza il modello Bedrock embeddings
        embeddings_model = BedrockEmbeddings()

    return model, embeddings_model
