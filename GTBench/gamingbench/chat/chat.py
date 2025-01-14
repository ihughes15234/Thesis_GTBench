
import os
from langchain_community.chat_models import ChatAnyscale
from langchain_community.llms import DeepInfra
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_vertexai import ChatVertexAI
from transformers import AutoTokenizer
from langchain_openai import ChatOpenAI
import time
import pandas as pd 

import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel
)

from typing import Dict, List, Union

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_custom_trained_model_sample(
    prompt: str,
    project: str ="", #add to run custom models

    endpoint_id: str = "", #add to run custom models
    location: str = "us-central1"
):

    api_endpoint = f"{location}-aiplatform.googleapis.com"
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    max_tokens = 20 # @param {type:"integer"}
    temperature = .2  # @param {type:"number"}
    top_p = 1.0  # @param {type:"number"}
    top_k = 2  # @param {type:"integer"}
    instances = [
    {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            },
        },
    ]
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances
    )
    response.predictions
    return(list(response.predictions))

def write_to_file(file_path, content):
    with open(file_path, 'w') as file:
        file.write(content)

def chat_llm(messages, model, temperature, max_tokens, n, timeout, stop, return_tokens=False, chat_seed=0):
    if model.__contains__("gpt"):
        iterated_query = False
        chat = ChatOpenAI(model_name=model,
                          openai_api_key=os.environ['OPENAI_API_KEY'],
                          temperature=temperature,
                          max_tokens=max_tokens,
                          n=n,
                          request_timeout=timeout,
                          )
    elif 'Open-Orca/Mistral-7B-OpenOrca' == model:
        iterated_query = True
        chat = ChatAnyscale(temperature=temperature,
                            anyscale_api_key=os.environ['ANYSCALE_API_KEY'],
                            max_tokens=max_tokens,
                            n=1,
                            model_name=model,
                            request_timeout=timeout)
    elif 'gemma_7b' == model:
        iterated_query = False
        chat = ChatVertexAI(temperature=temperature,
                            endpoint_id=endpoint_id,
                            project=project,
                            location=location,
                            max_tokens=max_tokens,
                            n=1,
                            request_timeout=timeout)
    else:
        # deepinfra
        iterated_query = True
        chat = ChatOpenAI(model_name=model,
                          openai_api_key=os.environ['DEEPINFRA_API_KEY'],
                          temperature=temperature,
                          max_tokens=max_tokens,
                          n=1,
                          request_timeout=timeout,
                          openai_api_base="https://api.deepinfra.com/v1/openai")

    longchain_msgs = []
    for msg in messages:
        if msg['role'] == 'system':
            longchain_msgs.append(SystemMessage(content=msg['content']))
        elif msg['role'] == 'user':
            longchain_msgs.append(HumanMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            longchain_msgs.append(AIMessage(content=msg['content']))
        else:
            raise NotImplementedError
    if n > 1 and iterated_query:
        #print('inner')
        response_list = []
        total_completion_tokens = 0
        total_prompt_tokens = 0
        for n in range(n):
            generations = chat.generate([longchain_msgs], stop=[
                stop] if stop is not None else None)
            responses = [
                chat_gen.message.content for chat_gen in generations.generations[0]]
            response_list.append(responses[0])
            completion_tokens = 1#generations.llm_output['token_usage']['completion_tokens']
            prompt_tokens = 1#generations.llm_output['token_usage']['prompt_tokens']
            total_completion_tokens += completion_tokens
            total_prompt_tokens += prompt_tokens
        responses = response_list
        completion_tokens = total_completion_tokens
        prompt_tokens = total_prompt_tokens
    else:
        if model.__contains__("phi"):
            string_tokenizer = "microsoft/Phi-3.5-mini-instruct" # ihughes15234/llama_3_1_8bi_tictactoe_dpo5epoch microsoft/Phi-3.5-mini-instruct "ihughes15234/llama_3_1_8bi_3k_each_1epoch" #meta-llama/Meta-Llama-3.1-8B-Instruct #"microsoft/Phi-3-mini-4k-instruct" #ihughes15234/phi_3_mini_500_combined "microsoft/Phi-3.5-mini-instruct"
            tokenizer = AutoTokenizer.from_pretrained(string_tokenizer, trust_remote_code=True)
            user_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            responses = predict_custom_trained_model_sample(prompt = user_prompt)
        elif model.__contains__("llama"):
            string_tokenizer = "meta-llama/Meta-Llama-3.1-8B-Instruct" # ihughes15234/llama_3_1_8bi_tictactoe_dpo5epoch microsoft/Phi-3.5-mini-instruct "ihughes15234/llama_3_1_8bi_3k_each_1epoch" #meta-llama/Meta-Llama-3.1-8B-Instruct #"microsoft/Phi-3-mini-4k-instruct" #ihughes15234/phi_3_mini_500_combined "microsoft/Phi-3.5-mini-instruct"
            tokenizer = AutoTokenizer.from_pretrained(string_tokenizer, trust_remote_code=True)
            user_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            responses = predict_custom_trained_model_sample(prompt = user_prompt)
        elif model.__contains__("gemini"):
            vertexai.init(project='', location="northamerica-northeast1") #add project to run custom models
            # europe-west1 europe-west2 europe-west3  europe-west4 europe-west8   us-east1  us-west1 europe-central2 asia-south1  'australia-southeast1'
            # southamerica-east1 asia-east1 northamerica-northeast1
            if model.__contains__("flash"):
                model_name = "gemini-1.5-flash-002"
            else:
                model_name = "gemini-1.5-pro-002"
            
            chat = GenerativeModel(
                model_name,
                system_instruction=[messages[0]['content']],
            )

            # # Set model parameters
            generation_config = GenerationConfig(
                temperature=0.2,
                top_p=1.0,
                top_k=2,
                max_output_tokens=20,
            )

            # # Set contents to send to the model
            contents = messages[1]['content'] #+ '\nResponse:'

            gen_model = chat.generate_content(
                contents,
                generation_config=generation_config
                ).text
            responses = [gen_model]
            time.sleep(30) #in case request per minute requests are low

        else:
            generations = chat.generate([longchain_msgs], stop=[
                stop] if stop is not None else None)
            responses = [
                chat_gen.message.content for chat_gen in generations.generations[0]]


        #below modified as this is only available for openAI style api
        completion_tokens = 1 #generations.llm_output['token_usage']['completion_tokens']
        prompt_tokens = 1 #generations.llm_output['token_usage']['prompt_tokens']

    return {
        'generations': responses,
        'completion_tokens': completion_tokens,
        'prompt_tokens': prompt_tokens
    }
