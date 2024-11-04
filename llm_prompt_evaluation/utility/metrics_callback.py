import json
import random
from difflib import SequenceMatcher
from langchain.callbacks.base import BaseCallbackHandler
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import util
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from scipy.stats import entropy
from llm_prompt_evaluation.config import *


class MetricsCallback(BaseCallbackHandler):

    def __init__(self, dataset, logger, embeddings=None):
        self.dataset = dataset
        self.embeddings = embeddings
        self.response_times = []  # Tempi di risposta
        self.success_responses = []  # Risposte generate
        self.error_responses = []  # Risposte errate
        self.error_responses_model = []  # Risposte errate del modello
        self.current_index = -1  # Input originali
        self.user_feedbacks = []  # Feedback degli utenti
        self.input_token_counts = []  # Conteggio dei token degli input
        self.output_token_counts = []  # Conteggio dei token delle risposte
        self.relevance_scores = []  # Pertinenza delle risposte
        self.vectorizer = TfidfVectorizer()  # Inizializzazione del vettorizzatore TF-IDF
        self.success_similarity_count = 0  # Conteggio del numero totale di risposte per similarità con successo
        self.error_similarity_count = 0  # Conteggio del numero totale di risposte per similarità con errore
        self.embedding_relevance_success = 0  # Conteggio del numero totale di risposte per similarità vettorializzata con successo
        self.blue_score_success = 0  # Conteggio del numero totale di blue score con successo
        self.rouge1_score_success = 0  # Conteggio del numero totale di rouge1 score con successo
        self.rouge2_score_success = 0  # Conteggio del numero totale di rouge2 score con successo
        self.rougeL_score_success = 0  # Conteggio del numero totale di rougeL score con successo
        self.entropy_success = 0  # Conteggio del numero totale dell'entropia con successo
        self.start_time = None  # inizio richiesta
        self.logger = logger

    def on_llm_start(self, graph, prompt, **kwargs):
        self.start_time = time.time()  # Inizio della misurazione del tempo
        self.current_index = kwargs.get("metadata", {}).get("index", -1)

    def on_llm_end(self, response, **kwargs):
        try:
            elapsed_time = time.time() - self.start_time  # Tempo impiegato
            message = response.generations[0][0].message
            # gestione del body
            if '"body":' not in message.content and '"error":' not in message.content:
                original_response = message.content
            else:
                content = json.loads(message.content
                                     .replace("\n", "").replace("<template>", "")
                                     .replace("</template>", "").replace("<response>", "")
                                     .replace("</response>", ""))
                if content["body"]:
                    original_response = content["body"]
                elif content["error"]:
                    original_response = content["error"]
                else:
                    original_response = content["general"]

            original_response = original_response.replace("\\", " ")

            self.response_times.append((elapsed_time, self.current_index))

            # conteggio dei token
            self.input_token_counts.append(message.usage_metadata["input_tokens"])
            self.output_token_counts.append(message.usage_metadata["output_tokens"])

            # Risposta attesa
            expected_response = self.get_expected_response()

            # Calcola la pertinenza della risposta mediante la similarità del coseno
            similarity_score = self.calculate_cosine_similarity(original_response, expected_response)
            self.success_similarity_count += 1 if similarity_score > SIMILARITY_THRESHOLD else 0
            self.error_similarity_count += 1 if similarity_score <= SIMILARITY_THRESHOLD else 0

            # calcola blue score
            blue_score_result = self.calculate_blue(original_response, expected_response)
            # calcola rouge score
            rouge_score_result = self.calculate_rouge(original_response, expected_response)
            # calcola embedding relevance
            if self.embeddings is not None:
                embedding_relevance_result = self.calculate_embedding_relevance(original_response, expected_response)
                self.embedding_relevance_success += 1 if embedding_relevance_result > SIMILARITY_EMBEDDING_THRESHOLD else 0
            # calcola blue and rouge score success
            self.blue_score_success += 1 if blue_score_result > BLUE_THRESHOLD else 0
            self.rouge1_score_success += 1 if rouge_score_result["rouge1"].fmeasure > ROUGE_THRESHOLD else 0
            self.rouge2_score_success += 1 if rouge_score_result["rouge2"].fmeasure > ROUGE_THRESHOLD else 0
            self.rougeL_score_success += 1 if rouge_score_result["rougeL"].fmeasure > ROUGE_THRESHOLD else 0
            # calcola l'entropia della risposta
            response_entropy = self.calculate_response_entropy(original_response)
            self.entropy_success += 1 if 0 <= response_entropy < 2.5 else 0

            # Calcola la pertinenza della risposta mediante la risposta di success similarity
            is_relevant = True if similarity_score > SIMILARITY_THRESHOLD else False
            self.relevance_scores.append(is_relevant)

            if not is_relevant:
                self.error_responses.append((original_response, self.current_index))
            else:
                self.success_responses.append((original_response, self.current_index))

        except Exception as e:
            self.logger.error(f"Error in on_llm_end: {e}")
            self.error_responses_model.append((response.generations[0][0].message.content, self.current_index))

    def get_expected_response(self):
        if self.current_index != -1:
            return self.dataset.iloc[self.current_index]["Risposta SQL Attesa"]
        else:
            return ""

    def calculate_cosine_similarity(self, response, expected_response):
        # Concatena le risposte per il calcolo della similarità
        documents = [response, expected_response]
        tfidf_matrix = self.vectorizer.fit_transform(documents)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
        return similarity[0][0]  # Restituisce la similarità coseno

    def calculate_embedding_relevance(self, generated_output, expected_output):
        # Calcola gli embeddings delle due risposte
        embeddings1 = self.embeddings.embed_query(generated_output)
        embeddings2 = self.embeddings.embed_query(expected_output)

        # Calcola la similarità del coseno
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
        return cosine_similarity

    @staticmethod
    def calculate_blue(generated_output, expected_output):
        # Tokenizza le frasi per BLEU
        reference = [
            nltk.word_tokenize(expected_output.lower(), preserve_line=True)]  # La risposta attesa è la reference
        candidate = nltk.word_tokenize(generated_output.lower(),
                                       preserve_line=True)  # La risposta generata è il candidato

        # Calcola il BLEU score
        smoothing_func = SmoothingFunction().method1
        bleu_score = sentence_bleu(reference, candidate, smoothing_function=smoothing_func)
        return bleu_score

    def calculate_rouge(self, generated_output, expected_output):
        # Inizializza lo scorer ROUGE
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(expected_output, generated_output)
        return scores

    @staticmethod
    def calculate_std_across_runs(llm_chain, input: dict, config: dict, n_runs=5):
        responses = []

        for _ in range(n_runs):
            result = llm_chain.invoke(input, config=config)
            response_text = json.loads(result.content)["body"]
            responses.append(response_text)

        # Calcola la lunghezza delle risposte per verificare la variabilità
        response_lengths = np.array([len(resp.split()) for resp in responses])

        # Calcola la deviazione standard delle lunghezze delle risposte
        std_dev = np.std(response_lengths)

        return responses, std_dev, True if 5 <= std_dev < 15 else False

    def calculate_response_entropy(self, response):
        # Conta la frequenza di ogni parola nella risposta
        words = response.split()
        word_counts = np.unique(words, return_counts=True)[1]

        # Calcola la probabilità di ogni parola
        word_probs = word_counts / len(words)

        # Calcola l'entropia usando la distribuzione di probabilità delle parole
        result = entropy(word_probs)
        return result

    # Funzione per introdurre errori ortografici in una frase
    @staticmethod
    def introduce_typo(text, error_rate=0.1):
        text_list = list(text)
        for i in range(len(text_list)):
            if random.random() < error_rate:
                # Sostituisci un carattere casuale
                text_list[i] = random.choice('abcdefghijklmnopqrstuvwxyz')
        return ''.join(text_list)

    # Funzione per calcolare l'accuracy robusta
    @staticmethod
    def calculate_robust_accuracy(llm_chain, prompt, config, original_output, expected_response):
        # Risultato originale
        is_original_correct = SequenceMatcher(None, original_output, expected_response).ratio() > 0.8

        # Risultato con perturbazione
        perturbed_prompt = MetricsCallback.introduce_typo(prompt)
        perturbed_output = llm_chain.invoke(perturbed_prompt, config=config)
        perturbed_output = json.loads(perturbed_output.content)["body"]
        is_perturbed_correct = SequenceMatcher(None, perturbed_output, expected_response).ratio() > 0.8

        # Calcolo dell'accuratezza robusta
        return is_original_correct, is_perturbed_correct

    @staticmethod
    def calculate_retention_rate(llm_chain, config, context_statements, follow_up_questions, expected_answers):
        correct_retentions = 0
        total_contexts = len(context_statements)
        # Poi, verifica se il modello ricorda con domande di follow-up
        for i, question in enumerate(follow_up_questions):
            response = llm_chain.invoke(question, config=config)
            response = json.loads(response.content)["body"]
            if response.lower() == expected_answers[i].lower():
                correct_retentions += 1

        # Calcola il tasso di mantenimento
        retention_rate = (correct_retentions / total_contexts) * 100
        return retention_rate

    def log_metrics(self):
        average_time = np.mean(
            [response_times[0] for response_times in self.response_times]) if self.response_times else 0
        total_time = np.sum([response_times[0] for response_times in self.response_times]) if self.response_times else 0
        max_time = np.max([response_times[0] for response_times in self.response_times]) if self.response_times else 0
        total_input_tokens = sum(self.input_token_counts)
        total_output_tokens = sum(self.output_token_counts)
        total_tokens = total_input_tokens + total_output_tokens
        average_tokens = np.mean(
            self.input_token_counts + self.output_token_counts) if self.input_token_counts and self.output_token_counts else 0
        accuracy = accuracy_score([1 if r else 0 for r in self.relevance_scores], [1] * len(self.relevance_scores))
        total_response = len(self.success_responses) + len(self.error_responses) + len(self.error_responses_model)

        self.logger.info("\n\nMetrics:\n")
        self.logger.info("Average Response Time (seconds): {:.2f}".format(average_time))
        self.logger.info("Max Response Time (seconds): {:.2f}".format(max_time))
        self.logger.info("Total Time (seconds): {:.2f}".format(total_time))
        self.logger.info("Average Tokens: {}".format(average_tokens))
        self.logger.info("Total Input Tokens: {} ".format(total_input_tokens))
        self.logger.info("Total Output Tokens: {}".format(total_output_tokens))
        self.logger.info("Total Tokens: {}".format(total_tokens))
        self.logger.info("Total cost (token): {:.2f}".format(((total_input_tokens * INPUT_COST_PER_TOKEN) +
                                                              (total_output_tokens * OUTPUT_COST_PER_TOKEN)) / 1000))
        self.logger.info("Total Response: {}".format(total_response))
        self.logger.info("Success Response Count: {}".format(len(self.success_responses)))
        self.logger.info("Error Response Count: {}".format(len(self.error_responses)))
        self.logger.info("Error Response Model Count: {}".format(len(self.error_responses_model)))
        self.logger.info("Success Responses Percentage: {:.2f}".format(len(self.success_responses) / total_response * 100))
        self.logger.info(
            "Error Responses Model Percentage: {:.2f}".format(len(self.error_responses_model) / total_response * 100))
        self.logger.info("Success Responses: {}".format(self.success_responses))
        self.logger.info("Error Responses: {}".format(self.error_responses))
        self.logger.info("Error Responses Model: {}".format(self.error_responses_model))
        self.logger.info("Success similarity Count: {}".format(self.success_similarity_count))
        self.logger.info("Error similarity Count: {}".format(self.error_similarity_count))
        if self.success_similarity_count + self.error_similarity_count > 0:
            self.logger.info("Success Similarity Percentage: {:.2f}".format(
                self.success_similarity_count / total_response * 100))
        else:
            self.logger.info("Success Similarity Percentage: 0")
        self.logger.info("Embedding Relevance Success: {}".format(self.embedding_relevance_success))
        self.logger.info("Embedding Relevance Error: {}".format(total_response - self.embedding_relevance_success))
        if self.success_similarity_count + self.error_similarity_count > 0:
            self.logger.info("Embedding Relevance Success Percentage: {:.2f}".format(
                self.embedding_relevance_success / total_response * 100))
        else:
            self.logger.info("Embedding Relevance Success Percentage: 0")
        self.logger.info("Blue Score Success: {}".format(self.blue_score_success))
        self.logger.info("Rouge1 Score Success: {}".format(self.rouge1_score_success))
        self.logger.info("Rouge2 Score Success: {}".format(self.rouge2_score_success))
        self.logger.info("RougeL Score Success: {}".format(self.rougeL_score_success))
        self.logger.info("Entropy Success: {}".format(self.entropy_success))
        self.logger.info("Overall Accuracy: {:.2f}".format(accuracy))

    def get_metrics(self):
        average_time = np.mean(
            [response_times[0] for response_times in self.response_times]) if self.response_times else 0
        total_time = np.sum([response_times[0] for response_times in self.response_times]) if self.response_times else 0
        max_time = np.max([response_times[0] for response_times in self.response_times]) if self.response_times else 0
        total_input_tokens = sum(self.input_token_counts)
        total_output_tokens = sum(self.output_token_counts)
        total_tokens = total_input_tokens + total_output_tokens
        average_tokens = np.mean(
            self.input_token_counts + self.output_token_counts) if self.input_token_counts and self.output_token_counts else 0
        accuracy = accuracy_score([1 if r else 0 for r in self.relevance_scores], [1] * len(self.relevance_scores))
        total_response = len(self.success_responses) + len(self.error_responses) + len(self.error_responses_model)

        metrics = {
            "average_time": round(average_time, 2),
            "max_time": round(max_time, 2),
            "total_time": round(total_time, 2),
            "average_tokens": round(average_tokens, 2),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "total_cost": round(((total_input_tokens * INPUT_COST_PER_TOKEN) + (
                        total_output_tokens * OUTPUT_COST_PER_TOKEN)) / 1000, 4),
            "total_response": total_response,
            "success_response_count": len(self.success_responses),
            "error_response_count": len(self.error_responses),
            "error_response_model_count": len(self.error_responses_model),
            "success_responses_percentage": round(len(self.success_responses) / total_response * 100, 2),
            "error_responses_model_percentage": round(len(self.error_responses_model) / total_response * 100, 2),
            "success_responses": self.success_responses,
            "error_responses": self.error_responses,
            "error_responses_model": self.error_responses,
            "success_similarity_count": self.success_similarity_count,
            "error_similarity_count": self.error_similarity_count,
            "success_similarity_percentage": round(self.success_similarity_count / total_response * 100, 2) if total_response > 0 else 0,
            "embedding_relevance_success": self.embedding_relevance_success,
            "embedding_relevance_error": total_response - self.embedding_relevance_success,
            "embedding_relevance_success_percentage": round(self.embedding_relevance_success / total_response * 100, 2) if total_response > 0 else 0,
            "blue_score_success": self.blue_score_success,
            "rouge1_score_success": self.rouge1_score_success,
            "rouge2_score_success": self.rouge2_score_success,
            "rougeL_score_success": self.rougeL_score_success,
            "entropy_success": self.entropy_success,
            "overall_accuracy": round(accuracy, 2)
        }

        return metrics
