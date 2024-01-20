"""LLM Module to Chatbot"""
import os
from typing import Union, List

from pinecone import Pinecone
from dotenv import load_dotenv
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.llamacpp import LlamaCpp
from sentence_transformers import SentenceTransformer

# Add key-value pair from .env file to environment variables
load_dotenv()

secret_key = os.getenv(
    key='PINECONE_API_KEY'
)

assert secret_key is not None, "Please provide your Pinecone's API Key"

index_name = os.getenv(
    key='PINECONE_INDEX_NAME'
)

assert index_name is not None, "Please provide your Pinecone's Index Name"

# Establish connection with Pinecone's index
pinecone_index = Pinecone(
    api_key=secret_key
).Index(
    name=index_name
)

# Establish callback to LLM for logging, monitoring, streaming and other tasks
callback_manager = CallbackManager(
    handlers=[StreamingStdOutCallbackHandler()]
)

def chat_model(
    temperature: float = 0.01
) -> LlamaCpp:
    """Chat Model Configuration"""
    return LlamaCpp(
        model_path='llama-2-7b-chat.gguf.q2_K.bin',
        callback_manager=callback_manager,
        verbose=True,
        temperature=temperature,
        max_tokens=1024,
        top_p=1,
        top_k=1,
        streaming=True
    )

def find_role(
    message: Union[SystemMessage, HumanMessage, AIMessage]
) -> str:
    """Identify role name from langchain.schema object"""
    if isinstance(message, SystemMessage):
        return 'system'

    if isinstance(message, HumanMessage):
        return 'user'

    if isinstance(message, AIMessage):
        return 'assistant'

    raise TypeError('Unknown message type.')

def convert_langchain_schema(
    messages: List[Union[HumanMessage, AIMessage]],
    window_context: int = 1 # Get n number of the last dialogue messages
) -> List[dict]:
    """Convert the langchain.schema format into a list of dictionary format."""
    return [
        {
            'role': find_role(message),
            'content': message.content
        } for message in messages[-(2 * window_context + 1):]
    ]

def llama2_prompt(
    messages: List[dict],
    context: str = ''
) -> str:
    """Convert messages in a list of dictionary format into Llama2 format"""
    # Create special tokens in Llama 2 prompt
    b_inst, e_inst = '[INST]', '[/INST]'
    b_sys, e_sys = '<<SYS>>\n', '<</SYS>>\n\n'
    bos, eos = '<s>', '</s>'

    system_prompt_list = [
        'You are a helpful and honest assistant.',
        'Always answer as accurate as possible.',
        'Use context below as a knowledge base.',
        'If else, use your own knowledge.'
        f'\n\nCONTEXT:\n{context}'
    ]

    default_system_prompt = ' '.join(system_prompt_list)

    # Embed system prompt at the beginning Llama 2 prompt
    if messages[0]['role'] != 'system':
        messages = [
            {
                'role': 'system',
                'content': default_system_prompt
            }
        ] + messages

    # Embed system and the first user prompt at the beginning
    messages = [
        {
            'role': messages[1]['role'],
            'content': b_sys + messages[0]['content'] + e_sys + messages[1]['content']
        }
    ] + messages[2:]

    # Iterate through all current dialogue messages between User and Assistant
    messages_list = [
        f"{bos}{b_inst} {(prompt['content'].strip())} {e_inst} {(answer['content'].strip())} {eos}"
        for prompt, answer in zip(messages[::2], messages[1::2])
    ]

    # Embed the last query from the User at the end of Llama 2 prompt
    messages_list.append(
        f"{bos}{b_inst} {(messages[-1]['content']).strip()} {e_inst}"
    )

    print(''.join(messages_list))
    return ''.join(messages_list)

def find_context(
    input_query: str
) -> str:
    """Do similarity search with external data indexed in Pinecone"""
    model = SentenceTransformer(
        model_name_or_path='all-MiniLM-L6-v2'
    )

    input_embedded = model.encode(
        sentences=input_query
    ).tolist()

    # Get the top 10 best relevant results based-on score
    results = pinecone_index.query(
        vector=input_embedded,
        top_k=10,
        include_metadata=True
    )['matches']

    context = ''

    # Filter the results based-on score with threshold of 0.5
    for result in results:
        if result['score'] > 0.5:
            context += result['metadata']['text'] + '\n\n'

    return context

def query_history(
    messages: List[Union[HumanMessage, AIMessage]],
    window_context: int = 1 # Get n number of the last User query
) -> str:
    """Get historical User queries for similarity search"""
    query = ''

    for message in messages[-(window_context + 1):]:
        if isinstance(message, HumanMessage):
            query += message.content.strip() + ' '

    return query

def get_answer(
    llm,
    messages
) -> str:
    """Get answer after query entered"""
    # Get the relevant context based-on User query
    context_found = find_context(
        input_query=query_history(
            messages=messages,
            window_context=1
        )
    )

    return llm.invoke(
        input=llama2_prompt(
            messages=convert_langchain_schema(
                messages=messages,
                window_context=0 if context_found != '' else 1
            ),
            context=context_found
        )
    )
