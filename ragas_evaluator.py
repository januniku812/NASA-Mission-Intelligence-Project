import os 

import sys
import types
import logging 
import numpy as np
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ragas_evaluator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# module injections to workaround cannot VertexAI from langchain_community.llms error
#classes exist within modules so we can create a mock module for the langchain_community.chat_models.vertexai
workaround_model = types.ModuleType("langchain_community.chat_models.vertexai")
class ChatVertexAI():
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VertexAI not supported in this environment"
        )

workaround_model.ChatVertexAI = ChatVertexAI 

sys.modules["langchain_community.chat_models.vertexai"] = workaround_model # patches the issue langchain_community.chat_models.vertexai issue before RAGAS is ever imported 
if "langchain_community.chat_models.vertexai" in sys.modules:
    logger.log(level=logging.INFO, msg="langchain_community.chat_models.vertexai mock module successfully injected")

llms_workaround_model = types.ModuleType("langchain_community.llms")
class VertexAI():
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "VertexAI not supported in this environment"
        )
llms_workaround_model.VertexAI = VertexAI 

sys.modules["langchain_community.llms"] = llms_workaround_model # patches the issue langchain_community.chat_models.vertexai issue before RAGAS is ever imported 
if "langchain_community.llms" in sys.modules:
    logger.log(level=logging.INFO, msg="langchain_community.llms mock module successfully injected")


# avoiding your system has an unsupported version of sqlite3. chroma requires sqlite3>=3.35.0 error 
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import ragas
from ragas.llms import LangchainLLMWrapper
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from ragas import EvaluationDataset 
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from ragas import SingleTurnSample
import traceback
import json
from langchain_classic.evaluation import load_evaluator, Criteria
# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import (
        faithfulness, 
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity,
        BleuScore,
        RougeScore
    )
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError as e: 
    traceback.print_exception(e)
    RAGAS_AVAILABLE = False


import rag_client # need to import rag_client class to mimic the document retrieval functionality and reconstruct contexts for system RAGAS evaluation
import llm_client # need to import llm_client to simulate RAG model answers in evaluating the efficiacy of the generation part of the RAG system 


class RAGASCompatibleEmbeddings(OpenAIEmbeddings):
    def embed_query(self, text: str):
        return self.embed_documents([text])[0]

def get_evaluation_configs(evaluation_type: str,
                           evaluator_llm: str = None, 
                           evaluator_embeddings: str = None): 
    """Returns list of pre-instantiated metrics to use in RAGAS evaluation based on evaluation type and evaluation LLM and evaluation embeddings models given"""
    
    if evaluation_type == "baseline":

        return [
            faithfulness,
            answer_relevancy
        ]
    elif evaluation_type == "batch_evaluation": 

        return [
            context_precision, 
            context_recall, 
            answer_similarity, 
            BleuScore(),
            RougeScore()
        ]


def create_evaluator_llm_embeddings(api_key: str, 
                                    evaluator_llm_model: str,
                                    evaluator_llm_embedding: str,
                                    dimensions: int = 51):
    """Returns instantiated InstructorLLM object and OpenAIEmbeddings object, connected to aysnchronous OpenAI client at api key given, for evaluation of RAG system"""
                                    
    if api_key.startswith("sk"):
        base_url = "https://api.openai.com/v1"
    elif api_key.startswith("voc"):
        base_url = "https://openai.vocareum.com/v1"
    else:
        base_url = os.getenv("OPENAI_BASE_URL")


    async_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    ) # synchronous open ai client

    evaluator_llm = llm_factory(
        evaluator_llm_model,
        client=async_client
    )

    # Create evaluator_embeddings with model test-embedding-3-small
    openai_embeddings = OpenAIEmbeddings(
        model=evaluator_llm_embedding, 
        api_key=api_key, 
        base_url=base_url
    )

    return evaluator_llm, openai_embeddings


def evaluate_response_quality(question: str, answer: str, contexts: List[str], evaluator_model: str, evaluator_llm_embedding: str, evaluation_type: str = "baseline") -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    try:
        # Create evaluator LLM with model gpt-3.5-turbo
        evaluator_llm, evaluator_embeddings = create_evaluator_llm_embeddings(os.getenv("OPENAI_API_KEY"), evaluator_model, evaluator_llm_embedding, 51)

        metrics = metrics = get_evaluation_configs("baseline")

        # Define an instance for each metric to evaluate one instance consists of a user input, retrieved context, and response/answer 

        ragas_dataset = EvaluationDataset.from_list([
            {
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts
            }
        ])


        # Evaluate the response using the metrics
        results = evaluate(
            dataset=ragas_dataset, 
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings
        ) # returns a dictionary object with metrics as keys and the metric evaluation floats as the respective values
        # Return the evaluation results --> need to complete
        results_dict = {metric.name: float(results[metric.name][0]) for metric in metrics}

        return results_dict

    except Exception as e:
        traceback.print_exception(e)
        return {metric.name: "inconclusive" for metric in metrics}


def test_evaluation(openai_key: str, 
                    dataset_path: str, # path to test questions formatted in json
                    test_questions_mission_category: str, 
                    top_k: str, # number of documents to consider from the retrieve_documents functionality
                    evaluator_llm_model: str, 
                    evaluator_llm_embedding_model: str, # model used in RAG system to be evaluated via llm_client (uses custom prompt) --> should match the model being used in the RAG system 
                    output_path: str,
                    generation_model: str):
    """Loads test question from json file for batch evaluation of RAG system using RAGAS metrics"""


    test_dataset = pd.read_json(dataset_path)

    available_backends = rag_client.discover_chroma_backends()

    mapping = {value["collection_name"]: key for key, value in available_backends.items()}

    selected_backend = available_backends[mapping[test_questions_mission_category]]

    # initialize rag database to get access to a collection 
    collection, success, error = rag_client.initialize_rag_system(
        selected_backend["directory"],
        selected_backend["collection_name"]
    )

    samples = []
    for test_entry in test_dataset[test_questions_mission_category]:
        question = test_entry.get("test_question")
        reference = test_entry.get("expected_answer") # the ideal reference answer used to evaluate the model's efficacy via the metrics defined 

        # simulating the retrieval section of the RAG system
        documents = rag_client.retrieve_documents(
            collection,
            question,
            top_k 
        )

        test_evaluation_retrieved_contexts = documents["documents"][0] 
        metadata_retrieved_contexts = documents["metadatas"][0]
        distances = documents["distances"][0]
        scores = [(1/(1+distance)) for distance in distances]



        # formatting context from retrieval section to generate answer for test_question --> bridge between retrieval and generation section 
        formatted_contexts = rag_client.format_context(test_evaluation_retrieved_contexts, metadata_retrieved_contexts, scores)

        # generating LLM response using llm_client (which integrates the custom prompt template) 
        response = llm_client.generate_response(
            openai_key,
            question, 
            formatted_contexts,
            None,
            generation_model
        )


        samples.append(
            SingleTurnSample(
                user_input=question,
                response=response, 
                retrieved_contexts=test_evaluation_retrieved_contexts,  # wil be used to evaluate retrieval efficiency
                reference=reference # will be used to evaluate generation efficiency
            )
        )

    evaluation_dataset = EvaluationDataset(
        samples=samples
    )
    
    evaluator_llm, evaluator_embeddings = create_evaluator_llm_embeddings(os.getenv("OPENAI_API_KEY"), evaluator_llm_model, evaluator_llm_embedding_model, 51)

    metrics = get_evaluation_configs("batch_evaluation", evaluator_llm=evaluator_llm, evaluator_embeddings=evaluator_embeddings)

    result = evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm, # when using the pre-instantiated, singleton object metrics ragas handles automatic parameter injection to pass the evaluator llm and the evaluator embeddings into the context precision, context recall, etc evaluation metrics 
        embeddings=evaluator_embeddings
    )

    results_dataframe = result.to_pandas() 
    results_dataframe.to_html("results_dataframe.html")

    summary = {
        "per-question": [],
        "per-metric": []
    }

    with open(output_path, "w") as outfile: 
        # outputted summary per question
        for index, row in enumerate(results_dataframe.itertuples()): 
            block = (
                "\n"
                "------------------------------------------------\n"
                f"Summary - Question {index+1}: {row.user_input}\n"
                f"Context Precision Score: {row.context_precision}\n"
                f"Context Recall Score: {row.context_recall}\n"
                f"Answer Similarity Score: {row.answer_similarity}\n"
                f"Bleu Score: {row.bleu_score}\n"
                f"Rouge score: {row[8]}\n"
                "------------------------------------------------\n"
                "\n"
            )

            summary["per-question"].append(block)
            outfile.write(block)

        # aggregate per metric
        for metric in metrics:
            metric_str = metric.name
            average_value = np.mean([getattr(row, metric_str) if metric_str != "rouge_score" else row[8] for index, row in enumerate(results_dataframe.itertuples())])
            block = f"\nMetric: {metric_str.title()} | Average Score: {average_value}"
            summary["per-metric"].append(block)
            outfile.write(block)

    with open(output_path.replace('txt', 'json'), 'w', encoding='utf-8') as json_file:
        json.dump(summary, json_file, indent=4)


    logger.log(level=logging.INFO, msg=f"Succesfully saved RAGAS batch evaluation results from test dataset at {dataset_path} to output path at {output_path}")
    return summary 
# ragas main function is to be used for test evaluation on a given json file inputted
if __name__ == "__main__":
    import argparse
    argument_parser = argparse.ArgumentParser(description="RAGAS Test Questions Batch Evaluation")
    argument_parser.add_argument("--testopenai-key", default=os.getenv("OPENAI_API_KEY"))
    argument_parser.add_argument("--dataset-path", default="test_questions.json") # NOTE: reference to evaluate dataset file in evaluation flow 
    argument_parser.add_argument("--test-questions-mission-category", 
                        choices=["apollo11", "apollo13", "challenger"],
                        help="Choose the mission set of questions within the test_questions.json input file to be evaluated"
                    ) # only argument that needs to be defined and has no default 
    argument_parser.add_argument("--top-k", type=int, default=3) # the number of documents to be extracted from the retrieve_documents functionality 
    argument_parser.add_argument("--evaluator-llm", type=str, default="gpt-4o") # model to use via the llm_client to model the rag system --> should be same model that is generating the answers in the RAG system 
    argument_parser.add_argument("--evaluator-llm-embeddings", type=str, default="text-embedding-3-small")
    argument_parser.add_argument("--output-path", type=str, default="batch_evaluation_results.txt")
    argument_parser.add_argument("--generation-model", type=str, default="gpt-3.5-turbo")

    arguments = argument_parser.parse_args()

    test_evaluation(
                    arguments.testopenai_key, 
                    arguments.dataset_path,
                    arguments.test_questions_mission_category,
                    arguments.top_k, 
                    arguments.evaluator_llm, 
                    arguments.evaluator_llm_embeddings, 
                    arguments.output_path,
                    arguments.generation_model
                )

        