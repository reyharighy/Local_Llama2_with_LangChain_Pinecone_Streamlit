## Overview
This repository contains the code and resources to create a chatbot using Llama 2 as the Large Language Model, Pinecone as the Vector Store for efficient similarity search, and Streamlit for building the user interface.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)

## Introduction
This project aims to showcase the integration of technologies to build an intelligent and interactive chatbot. The main focus is to take advantage of Llama 2 as open source Large Language Model developed by Meta AI as introduced in their [website](https://ai.meta.com/llama/). It is available for free for research and commercial use. 

While building with Llama 2, this repository is intended to leverage its factual accuracy and consistency by providing it with reliable and up-to-date information from the knowledge base. Achieving this involves the model with an external database that could be used to store such novel information. In order to keep it as simple as it goes, this repository focuses solely on meeting the outlined objectives without delving into alternative technologies that would certainly complement the project but are outside its specific scope. 

## Features
- **LangChain**: A specialized framework designed for developping language model applications, providing seamless integration with Llama 2 model. The framework offers off-the-shelf chains for easy initiation as well as customizable components for tailoring existing chains or building new ones. The LangChain libraries themselves are made up of several different packages, with `langchain_community` serving as a hub for third party integrations. Inside of it, there's `llama-cpp-python` library that's useful this specific purpose of this repository. It acts as a Python binding for `llama.cpp`. It also supports inference for many Llama 2 models. In this case, we would provide the model to run locally using the quantized Llama 2 model of **llama-2-7b-chat** as one of its flavors.
- **Pinecone**: Leverage Pinecone as a Vector Store for efficient similarity search and retrieval of contextually relevant responses. Pinecone runs serverless that lets you deliver remarkable chatbot applications with a certain number of advantages. It has power to search across billions of embeddings with ultra-low query latency. You can get started for free, then upgrade and scale as needed. Let alone, no need to maintain infrastructure, monitor services, or troubleshoot algorithms.  
- **Streamlit**: Build a user-friendly interface using Streamlit, allowing users to interact seamlessly with the chatbot. No front-end experience required. It's the UI powering Large Language Model movement. That means GenAI and Streamlit: A perfect match.

## Requirements
To set up the project, follow these steps:
1. Create a dedicated directory for the purpose of this project using your preferred file manager or command-line interface.
2. It's always a best practice to create a virtual environment to manage dependencies for your project which you can decide to give the name arbitrarily. I personally name it `llm_env` in my project. Please make sure that you already have Python installed, preferably with a version above 3.8.x or higher in order to get along with this. Open the terminal and navigate to the project directory you have provided. Run this command to create a virtual environment.

    ```bash
    python -m venv llm_env 
    ```

2. Activate the virtual environment after it's done created.
    - On Windows:

    ```bash
    llm_env\Scripts\activate
    ```

    - On Unix or MacOS:

    ```bash
    source llm_env\bin\activate
    ```

3. Download `requirements.txt` file provided in this repository and navigate to the directory where the file is located. Then, install all dependencies included by running this command.

   ```bash
   pip install -r requirements.txt
   ```

   Libraries:
   
    ![PyPI - Version](https://img.shields.io/pypi/v/streamlit?style=for-the-badge&label=Streamlit&color=%236666ff)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/langchain?style=for-the-badge&label=langchain&color=%23ff3300)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/easygoogletranslate?style=for-the-badge&label=easygoogletranslate&color=%23ffcc00)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/pinecone-client?style=for-the-badge&label=pinecone-client&color=%2333cc33)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/python-dotenv?style=for-the-badge&label=python-dotenv&color=%23ff66cc)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/sentence-transformers?style=for-the-badge&label=sentence-transformers&color=%2333cccc)<br>
    ![PyPI - Version](https://img.shields.io/pypi/v/llama-cpp-python?style=for-the-badge&label=llama-cpp-python&color=%23ff00ff)<br>

4. Download **llama-2-7b-chat**
????
- Prepare the external data as knowledge base
- Set up .env file to connect to Pinecone
- Set up vector store on Pinecone using sentence transformers
- Change the number of context to 1024
