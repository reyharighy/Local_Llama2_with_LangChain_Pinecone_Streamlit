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
`python -m env langchain_env`
