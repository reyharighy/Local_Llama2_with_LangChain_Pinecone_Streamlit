# Local_Llama2_with_Pineconce_Streamlit

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

   Libraries:<br>
   ![Pandas License](https://img.shields.io/badge/pandas-1.4.2-lightgrey)<br>

   ```bash
   pip install -r requirements.txt
   ```

5. zzzzz
    
    ![Pandas License](https://img.shields.io/badge/numpy-1.23.2-yellow)
    ![Pandas License](https://img.shields.io/badge/seaborn-0.11.2-blue)
    ![Pandas License](https://img.shields.io/badge/matplotlib-3.5.1-red)<br>
    ![scikit-learn ](https://img.shields.io/badge/scikit--learn-1.2.2-coral?labelColor=grey&style=flat)<br>
    ![imblearn ](https://img.shields.io/badge/imblearn-0.0-olive?labelColor=grey&style=flat)<br>
    ![category-encoders ](https://img.shields.io/badge/category--encoders-2.6.0-emerald?labelColor=grey&style=flat)<br>
    ![lightgbm ](https://img.shields.io/badge/lightgbm-3.3.5-pink?labelColor=grey&style=flat)<br>
    ![xgboost](https://img.shields.io/badge/xgboost-1.7.5-navy?labelColor=grey&style=flat)<br>
