import streamlit as st
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, \
    HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
import config as conf
from utility.metrics_callback import MetricsCallback
from utility import utils as ut
from utility import metrics_callback as mp
from utility import templates as te
from utility import sessions as ss
from streamlit_chartjs.st_chart_component import st_chartjs

# Inizializza il logger
logger = ut.setup_logger("LLM Prompt Evaluation")

# Titolo principale
st.title("Dashboard Verifica Prompt LLM")

# Area per l'upload del CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
df = None

# Se un file è stato caricato
if uploaded_file is not None:
    try:
        df = ut.load_dataset_from_csv(uploaded_file)
        df["Risposta SQL Attesa"] = df["Richiesta Utente"]
        df["Richiesta Utente"] = df.index
        df.reset_index(drop=True, inplace=True)
        st.write(df)  # Visualizza i dati caricati
        st.session_state.button_disabled = False

    except Exception as e:
        logger.error("Errore nel caricamento del file CSV: {}".format(e))
        st.error("Errore nel caricamento del file CSV.")


def pie_chart(success_percentage, error_percentage, title):
    pie_chart_data = {
        "labels": ["Success", "Error"],
        "datasets": [
            {
                "label": "Percentage",
                "data": [success_percentage, error_percentage],
                "backgroundColor": ["#e74c3c", "#28b463"]
            }
        ],
    }

    st_chartjs(data=pie_chart_data, chart_type="pie", title=title, legend_position="top")


def generate_results(metrics_callback: MetricsCallback, total_time_col, average_time_col, max_time_col, total_token_col,
                     total_input_token_col, total_output_token_col, total_response_col, success_response_col,
                     error_response_col, error_response_model_col, success_similarity_col, error_similarity_col,
                     embedding_relevance_col, blue_score_col, rouge1_score_col, rouge2_score_col, rougeL_score_col,
                     overall_accuracy_col, success_responses_percentage_col, error_responses_model_percentage,
                     success_similarity_percentage, embedding_relevance_success_percentage):
    """
    Questa funzione genera i grafici per le metriche.

    :param metrics_callback: callback delle metriche
    """
    metrics_callback.log_metrics()
    metrics = metrics_callback.get_metrics()

    with total_time_col:
        # Totale tempo di esecuzione
        st.markdown(
            "<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>TOTALE TEMPO DI ESECUZIONE</h3>",
            unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['total_time']}</h1>", unsafe_allow_html=True)
    with average_time_col:
        # Tempo medio di esecuzione
        st.markdown(
            "<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>TEMPO MEDIO DI ESECUZIONE</h3>",
            unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['average_time']}</h1>", unsafe_allow_html=True)
    with max_time_col:
        # Tempo massimo di esecuzione
        st.markdown(
            "<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>TEMPO MASSIMO DI ESECUZIONE</h3>",
            unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['max_time']}</h1>", unsafe_allow_html=True)
    with total_token_col:
        # Totale token
        st.markdown("<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>TOTALE TOKEN</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['total_tokens']}</h1>", unsafe_allow_html=True)
    with total_input_token_col:
        # Totale token di input
        st.markdown(
            "<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>TOTALE TOKEN DI INPUT</h3>",
            unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['total_input_tokens']}</h1>", unsafe_allow_html=True)
    with total_output_token_col:
        # Totale token di output
        st.markdown(
            "<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>TOTALE TOKEN DI OUTPUT</h3>",
            unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['total_output_tokens']}</h1>", unsafe_allow_html=True)
    with total_response_col:
        # Totale risposte
        st.markdown("<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>TOTALE RISPOSTE</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['total_response']}</h1>", unsafe_allow_html=True)
    with success_response_col:
        # Risposte corrette
        st.markdown("<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>RISULTATI CORRETTI</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['success_response_count']}</h1>", unsafe_allow_html=True)
    with error_response_col:
        # Risposte errate
        st.markdown("<h3 style='text-align: center; border: 2px solid red; padding: 10px;'>RISULTATI ERRATI</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['error_response_count']}</h1>", unsafe_allow_html=True)
    with error_response_model_col:
        # Risposte errate modello
        st.markdown(
            "<h3 style='text-align: center; border: 2px solid red; padding: 10px;'>RISULTATI ERRATI MODELLO</h3>",
            unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['error_response_model_count']}</h1>",
                    unsafe_allow_html=True)
    with success_similarity_col:
        # Risposte corrette simili
        st.markdown(
            "<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>RISPOSTE CORRETTE SIMILI</h3>",
            unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['success_similarity_count']}</h1>",
                    unsafe_allow_html=True)
    with error_similarity_col:
        # Risposte errate simili
        st.markdown("<h3 style='text-align: center; border: 2px solid red; padding: 10px;'>RISPOSTE ERRATE SIMILI</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['error_similarity_count']}</h1>", unsafe_allow_html=True)
    with embedding_relevance_col:
        # Risposte rilevanti
        st.markdown("<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>RISPOSTE RILEVANTI</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['embedding_relevance_success']}</h1>",
                    unsafe_allow_html=True)
    with blue_score_col:
        # Blue score
        st.markdown("<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>BLUE SCORE</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['blue_score_success']}</h1>", unsafe_allow_html=True)
    with rouge1_score_col:
        # Rouge1 score
        st.markdown("<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>ROUGE1 SCORE</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['rouge1_score_success']}</h1>", unsafe_allow_html=True)
    with rouge2_score_col:
        # Rouge2 score
        st.markdown("<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>ROUGE2 SCORE</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['rouge2_score_success']}</h1>", unsafe_allow_html=True)
    with rougeL_score_col:
        # RougeL score
        st.markdown("<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>ROUGEL SCORE</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['rougeL_score_success']}</h1>", unsafe_allow_html=True)
    with overall_accuracy_col:
        # Overall accuracy
        st.markdown("<h3 style='text-align: center; border: 2px solid green; padding: 10px;'>OVERALL ACCURACY</h3>",
                    unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center;'>{metrics['overall_accuracy']}</h1>", unsafe_allow_html=True)
    with success_responses_percentage_col:
        # Percentuale risposte corrette
        pie_chart(metrics['success_responses_percentage'], metrics['error_responses_model_percentage'],
                  "Percentuale Risposte Corrette")
    with error_responses_model_percentage:
        # Percentuale risposte errate
        pie_chart(metrics['error_responses_model_percentage'], metrics['success_responses_percentage'],
                  "Percentuale Risposte Errate")
    with success_similarity_percentage:
        # Percentuale risposte simili
        pie_chart(metrics['success_similarity_percentage'], 100 - metrics['success_similarity_percentage'],
                  "Percentuale Risposte Similarity")
    with embedding_relevance_success_percentage:
        # Percentuale risposte rilevanti
        pie_chart(metrics['embedding_relevance_success_percentage'],
                  100 - metrics['embedding_relevance_success_percentage'], "Percentuale Risposte Embedding Similarity")


def reload_header_data(metrics_callback, col1_place_success, col2_place_error, col3_place_total):
    """
    Ricarica i dati dell'header.

    :param metrics_callback: callback delle metriche
    :param col1_place_success: success_responses
    :param col2_place_error: error_responses
    :param col3_place_total: total_responses
    :return:
    """
    col1_place_success.write(f"<div style='border: 2px solid red; padding: 10px; text-align: center;'><h4>TOTALE NUMERO RISPOSTE ERRATE</h4><h1 style='color: red;'>{len(metrics_callback.success_responses)}</h1></div>",
                             unsafe_allow_html=True)
    col2_place_error.write(f"<div style='border: 2px solid green; padding: 10px; text-align: center;'><h4>TOTALE NUMERO RISPOSTE CORRETTE</h4><h1 style='color: green;'>{len(metrics_callback.error_responses)}</h1></div>",
                           unsafe_allow_html=True)
    col3_place_total.write(
        f"<div style='border: 2px solid gray; padding: 10px; text-align: center;'><h4>TOTALE NUMERO RISPOSTE ANALIZZATE</h4><h1 style='color: gray;'>{len(metrics_callback.success_responses) + len(metrics_callback.error_responses)}</h1></div>",
        unsafe_allow_html=True)


def main():
    # Creazione di 3 colonne per i risultati
    col1, col2, col3 = st.columns(3)
    col1_place_success, col2_place_error, col3_place_total = st.empty(), st.empty(), st.empty()

    # Creazione dei grafici
    total_time_col, average_time_col, max_time_col = st.columns(3)
    total_token_col, total_input_token_col, total_output_token_col = st.columns(3)
    total_response_col, success_response_col, error_response_col = st.columns(3)
    error_response_model_col, success_similarity_col, error_similarity_col = st.columns(3)
    embedding_relevance_col, blue_score_col, rouge1_score_col = st.columns(3)
    rouge2_score_col, rougel_score_col, overall_accuracy_col = st.columns(3)
    success_responses_percentage_col, error_responses_model_percentage = st.columns(2)
    success_similarity_percentage, embedding_relevance_success_percentage = st.columns(2)

    # TODO: Richieste/Risposte generate (una textarea vuota in questo caso)
    # st.text_area("Richieste/Risposte generate", "")

    # inizializza la callback delle metriche
    metric_callbacks = mp.MetricsCallback(df, logger)

    if st.button("Run", disabled=st.session_state.button_disabled):
        # Nasconde i risultati
        st.session_state.show_results = False
        # Disabilita il bottone
        st.session_state.button_disabled = True

        st.info("Elaborazione in corso...")

        # load model and embedding model
        model, embeddings_model = ut.routing_model(conf.MODEL, metric_callbacks=[metric_callbacks])

        # set embedding model for metrics
        metric_callbacks.embeddings = embeddings_model

        # Inizializza i system e human prompt
        system_prompt = (SystemMessagePromptTemplate
                         .from_template(template=te.SYSTEM_PROMPT_TEMPLATE %
                                                 (datetime.now().strftime("%Y-%m-%d")),
                                        partial_variables={"schema": ut.get_prompt_input(),
                                                           "template": te.JSON_TEMPLATE,
                                                           "examples": te.EXAMPLES_TEMPLATE}))
        human_prompt = HumanMessagePromptTemplate.from_template(template=te.HUMAN_TEMPLATE)
        # crea il prompt finale
        if conf.USE_HISTORY:
            prompt_template = ChatPromptTemplate.from_messages([system_prompt,
                                                                MessagesPlaceholder(variable_name="history"),
                                                                human_prompt])
        else:
            prompt_template = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

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
                                             config={"metadata": {"index": index},
                                                     "configurable": {"session_id": "123"}})
            logger.info("result: {}".format(response.content))
            reload_header_data(metric_callbacks, col1_place_success, col2_place_error, col3_place_total)

        st.session_state.show_results = True
        if st.session_state.show_results:
            # genera i risultati finali
            generate_results(metric_callbacks, total_time_col, average_time_col, max_time_col, total_token_col,
                             total_input_token_col, total_output_token_col, total_response_col, success_response_col,
                             error_response_col, error_response_model_col, success_similarity_col, error_similarity_col,
                             embedding_relevance_col, blue_score_col, rouge1_score_col, rouge2_score_col,
                             rougel_score_col,
                             overall_accuracy_col, success_responses_percentage_col, error_responses_model_percentage,
                             success_similarity_percentage, embedding_relevance_success_percentage)

        # Riabilita il bottone dopo l'attesa
        st.session_state.button_disabled = False

if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = True

if "show_results" not in st.session_state:
    st.session_state.show_results = False

main()
