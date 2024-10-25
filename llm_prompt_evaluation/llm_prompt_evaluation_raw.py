from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
import config as conf
from utility import utils as ut
from utility import metrics_callback as mp
from utility import templates as te
from utility import sessions as ss


def main():
    # setup logging
    logger = ut.setup_logger("LLM Prompt Evaluation")

    # load dataframe
    df = ut.load_dataset_from_csv("../documents/casi_d_uso_grafici_tabelle.csv")

    # inizializza la callback delle metriche
    metric_callbacks = mp.MetricsCallback(df, logger)

    # load model and embedding model
    model, embeddings_model = ut.routing_model(conf.MODEL, metric_callbacks=[metric_callbacks])

    # set embedding model for metrics
    metric_callbacks.embeddings = embeddings_model

    # Inizializza i system e human prompt
    system_prompt = (SystemMessagePromptTemplate
                     .from_template(template=te.SYSTEM_PROMPT_TEMPLATE %
                                             (datetime.now().strftime("%Y-%m-%d")),
                                    partial_variables={"schema": ut.get_prompt_input(), "template": te.JSON_TEMPLATE,
                                                       "examples": te.EXAMPLES_TEMPLATE}))
    human_prompt = HumanMessagePromptTemplate.from_template(template=te.HUMAN_TEMPLATE)
    # crea il prompt finale
    prompt_template = ChatPromptTemplate.from_messages([system_prompt,
                                                        MessagesPlaceholder(variable_name="history"),
                                                        human_prompt])

    # costituisce la catena di esecuzione
    llm_chain = prompt_template | model

    if conf.USE_HISTORY:
        # importante aggiungere la chiave history per la gestione della conversazione
        runnable_chain = RunnableWithMessageHistory(llm_chain, ss.get_session_history, input_messages_key="input",
                                                    history_messages_key="history")
    else:
        runnable_chain = llm_chain

    # Esegui il modello su ogni riga del DataFrame
    for index, row in df.iterrows():
        logger.info("request: {}".format(row["Richiesta Utente"]))
        response = runnable_chain.invoke({"input": row["Richiesta Utente"]},
                                         config={"metadata": {"index": index}, "configurable": {"session_id": "123"}})
        logger.info("result: {}".format(response.content))

    # Log delle metriche
    metric_callbacks.log_metrics()


if __name__ == '__main__':
    main()
