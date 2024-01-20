## Overview
This repository contains the code and resources to create a chatbot using Llama 2 as the Large Language Model, Pinecone as the Vector Store for efficient similarity search, and Streamlit for building the user interface.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Knowledge Base Improvement](#knowledge-base-improvement)

## Introduction
This project aims to showcase the integration of technologies to build an intelligent and interactive chatbot that runs locally. The main focus is to take advantage of the Llama 2 as open source Large Language Model developed by Meta AI as introduced in [their website](https://ai.meta.com/llama/).

While building with Llama 2, this repository is intended to leverage its factual accuracy and consistency by providing it with reliable and up-to-date information from the knowledge base. Achieving this involves the model with an external database that could be used to store such novel information. In order to keep it as simple as it goes, this repository focuses solely on meeting the outlined objectives without delving into alternative technologies that would certainly complement the project but are outside its specific scope. 

## Features
- **LangChain**: A specialized framework designed for developping language model applications, providing seamless integration with the powerful Llama 2 model. The framework offers off-the-shelf chains for easy initiation as well as customizable components for tailoring existing chains or building new ones. The LangChain libraries themselves are made up of several different packages, with `langchain_community` serving as a hub for third party integrations. Within this package, `llama-cpp-python` is particularly relevant for the specific purpose of this repository. It acts as a Python binding for `llama.cpp` and supports inference for many Llama 2 models.
- **Pinecone**: Leverage Pinecone as a Vector Store for efficient similarity search and retrieval of contextually relevant responses. Pinecone runs serverless that lets you deliver remarkable chatbot applications with a certain number of advantages. It has power to search across billions of embeddings with ultra-low query latency. You can get started for free, then upgrade and scale as needed.
- **Streamlit**: Build a user-friendly interface using Streamlit, allowing users to interact seamlessly with the chatbot. No front-end experience required. It's the UI powering Large Language Model movement. That means GenAI and Streamlit: A perfect match.

## Requirements
To set up the project, follow these steps:
1. Create a dedicated directory for the purpose of this project using your preferred file manager or command-line interface.
2. It's always a best practice to create a virtual environment to manage dependencies for your project which you can decide to give the name arbitrarily. I personally name it `llm_env` in my project. Please make sure that you already have Python installed, preferably with a version above 3.8.x or higher in order to get along with this. Open the terminal and navigate to the project directory you have provided. Run this command to create a virtual environment.

    ```bash
    python -m venv llm_env 
    ```

3. Activate the virtual environment after it's done created.
    - On Windows:

    ```bash
    llm_env\Scripts\activate
    ```

    - On Unix or MacOS:

    ```bash
    source llm_env\bin\activate
    ```

4. Download `requirements.txt` file provided in this repository and navigate to the directory where the file is located. Then, install all dependencies included by running this command.

   ```bash
   pip install -r requirements.txt
   ```

   Libraries:
   
    ![PyPI - Version](https://img.shields.io/pypi/v/unstructured%20%5Ball-docs%5D?style=for-the-badge&label=unstructured%20%5Ball-docs%5D&color=%23ffffff)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/langchain?style=for-the-badge&label=langchain&color=%23ff9933)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/streamlit?style=for-the-badge&label=Streamlit&color=%236666ff)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/easygoogletranslate?style=for-the-badge&label=easygoogletranslate&color=%23ffcc00)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/pinecone-client?style=for-the-badge&label=pinecone-client&color=%2333cc33)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/python-dotenv?style=for-the-badge&label=python-dotenv&color=%23ff66cc)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/sentence-transformers?style=for-the-badge&label=sentence-transformers&color=%2333cccc)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/llama-cpp-python?style=for-the-badge&label=llama-cpp-python&color=%23ff00ff)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/gguf?style=for-the-badge&label=gguf&color=%23ffacde)<br>

## Installation
Llama 2 comes with various flavors that could be regarded as a family of state-of-the-art open-access Large Language Model. It's available with 12 open-access models with detailed of 3 base models and 3 fine-tuned ones with the original Meta checkpoints, plus their corresponding transformers models. Please go find more information about it on [Hugging Face](https://huggingface.co/meta-llama). In order to install the Llama 2 model, you need to follow these steps:
1. Download one of GGML version model of Llama 2 on [Hugging Face Repository](https://huggingface.co/localmodels/Llama-2-7B-Chat-ggml/tree/main). I personally choose `llama-2-7b-chat.gguf.q2_K` version as the foundational model that runs locally. You can choose whichever model that you want.

    **Note**: GGML is a machine learning library designed specifically to handle large models efficiently and deliver high performance on standard hardware. It employs a quantized representation of model weights, meaning it utilizes approximated parameters instead of full version. While this may result in a slight reduction in accuracy, the significant trade-off is evident in the resource efficiency it offers. This makes GGML an ideal starting point for most local machines, particularly those not equipped with GPUs for machine learning or with limited RAM.

2. Once downloaded, the GGML version needs to be converted to GGUF as proposed in [this discussion](https://github.com/abetlen/llama-cpp-python/pull/633). This conversion is considered a breaking change to model files that work with `llama.cpp`. Download `convert-llama-ggml-to-gguf.py` file provided in this repository.
3. Move the `convert-llama-ggml-to-gguf.py` script directly to the directory where the GGML version model is located.
4. Run this command on command-line interface.

   ```bash
   python convert-llama-ggml-to-gguf.py --eps 1e-5 --input llama-2-7b-chat.ggmlv3.q2_K.bin --output llama-2-7b-chat.gguf.q2_K.bin
   ```

    **Note**: Change the name of GGML version model in the script input to the one you have downloaded. Optionally, specify an arbitrary name for the GGUF version model as the script output.
 
## Usage
The chatbot application in this repository is designed to behave as an intelligent and interactive assistant, providing insightful replies to a wide range of queries. Leveraging the capabilities of the Llama 2 open-access models, the chatbot aims to offer valuable assistance in various domains. By this, we should be able to know how to prompt the models as well as how to change the system prompt. You can actually get to know about it through this release of [Hugging Face Blog](https://huggingface.co/blog/llama2#how-to-prompt-llama-2). 

The prompt template for the first turn looks like this:

```text
<s>[INST] <<SYS>>
{{ system_prompt }}
<</SYS>>

{{ user_message }} [/INST]
```

This template follows the model's training procedure as described in [Llama 2 Paper](https://huggingface.co/papers/2307.09288). We can use any system prompt we want, but it's crucial that the format matches the one used during training. Simple usage of the Llama 2 model has been provided on this repository from [this notebook](simple_usage.ipynb).

## Knowledge Base Improvement
As outlined in the introduction section, the primary objective of this repository is to enhance the accuracy of the Llama 2 model when providing answers to questions related to contexts outside its training data. To achieve this, we integrate the Llama 2 model with Pinecone, serving as an index database to store additional information that may not be present in Llama 2's knowledge base.

As part of our research, we have included trending and hot topics related to Indonesia's 2024 general election on this repository from `external_data` folder. It's important to note that this information is beyond Llama 2's knowledge, and we leverage Pinecone to store and retrieve this curated content. The information provided is reliable and maintains a neutral stance, avoiding any tendencies toward specific political choices. This research serves a purely academic and exploratory purpose.

```text
Registration stages for presidential and vice-presidential candidates for 2024 Indonesia’s general election
ended on October 25th, 2023. The general election will be held simultaneously throughout Indonesia on
February 14th, 2024. There are three pairs of presidential and vice-presidential candidates having
registered with Komisi Pemilihan Umum (KPU). Here’s the list of candidate pairs:

Candidate pair number 1 are Anies Rasyid Baswedan as presidential candidate and Abdul Muhaimin
Iskandar as vice presidential candidate. They were promoted by Partai Nasional Demokrat (Nasdem),
Partai Kebangkitan Bangsa (PKB), and Partai Keadilan Sejahtera (PKS). They registered with KPU on
October 19th, 2023.

Candidate pair number 2 are Prabowo Subianto as presidential candidate and Gibran Rakabuming Raka
as vice presidential candidate. They were promoted by Partai Gerakan Indonesia Raya (Gerindra), Partai
Golongan Karya (Golkar), Partai Demokrat (Demokrat), Partai Amanat Nasional (PAN), Partai Solidaritas 
Indonesia (PSI), Partai Gelombang Rakyat Indonesia (Gelora), Partai Bulan Bintang (PBB), Partai Rakyat
Adil Makmur (Prima), and Partai Garda Perubahan Indonesia (Garuda). They registered with KPU on
October 19th, 2023.

Candidates number 3 are Ganjar Pranowo as presidential candidate and Mahfud MD as vice presidential
candidate. They were promoted by Partai Demokrasi Indonesia Perjuangan (PDIP), Partai Persatuan
Pembangunan (PPP), Partai Persatuan Indonesia (Perindo), and Partai Hati Nurani Rakyat (Hanura). They
registered with KPU on October 25th, 2023.
```
