"""Chatbot with Streamlit UI"""
import time
import streamlit
from langchain.schema import (HumanMessage, AIMessage)
from easygoogletranslate import EasyGoogleTranslate
from utils import (chat_model, get_answer)

def reset_messages() -> None:
    """Clear the messages"""
    # Establish session state for conversation in English and Indonesian
    streamlit.session_state.english = []
    streamlit.session_state.indonesian = []

    # Detect language used for each session
    streamlit.session_state.language_used = []

def initialize() -> str:
    """Initialize Page and Messages"""
    streamlit.set_page_config(
        page_title='Personal Chatbot'
    )

    # Get the value of chosen language at each session
    language = streamlit.sidebar.radio(
        label='Language',
        options=('English', 'Indonesian'),
        key='language'
    )

    # Decorate the text based-on the chosen language
    page_text = {
        'English': ['Personal Chatbot', 'Options', 'Clear Conversation'],
        'Indonesian': ['Chatbot Pribadi', 'Opsi', 'Hapus Percakapan']
    }

    streamlit.header(
        body=page_text[language][0]
    )

    streamlit.sidebar.title(
        body=page_text[language][1]
    )

    clear_button = streamlit.sidebar.button(
        label=page_text[language][2]
    )

    # Reset the messages if clear button is pressed
    if clear_button or 'english' not in streamlit.session_state:
        reset_messages()

    streamlit.session_state.language_used.append(
        language
    )

    # Reset the messages if language used changes
    if len(set(streamlit.session_state.language_used)) > 1:
        reset_messages()

    return language

def main() -> None:
    """Running app.py"""
    language = initialize()

    llm = chat_model()

    page_text = {
        'English': ['Message Personal Chatbot...', 'Chatbot is thinking...'],
        'Indonesian': ['Kirimkan Pesan...', 'Chatbot sedang berpikir...']
    }

    if user_input := streamlit.chat_input(
        placeholder=page_text[language][0]
    ):
        # Translate the query input if chosen language is Indonesian
        if language == 'Indonesian':
            processed_input = EasyGoogleTranslate(
                source_language='id',
                target_language='en'
            ).translate(
                text=user_input
            )
        else:
            processed_input = user_input

        streamlit.session_state.english.append(
            HumanMessage(
                content=processed_input
            )
        )

        streamlit.session_state.indonesian.append(
            HumanMessage(
                content=user_input
            )
        )

        # Get the current messages saved in session states
        markdown_text = {
            'Indonesian': streamlit.session_state.indonesian,
            'English': streamlit.session_state.english
        }

        for message in markdown_text[language]:
            if isinstance(message, HumanMessage):
                with streamlit.chat_message(
                    name='user'
                ):
                    streamlit.markdown(
                        body=message.content
                    )
            elif isinstance(message, AIMessage):
                with streamlit.chat_message(
                    name='assistant'
                ):
                    streamlit.markdown(
                        body=message.content
                    )

        # Running the model to get the answer in each session
        with streamlit.spinner(
            text=page_text[language][1]
        ):
            answer = get_answer(
                llm=llm,
                messages=streamlit.session_state.english
            )

        streamlit.session_state.english.append(
            AIMessage(
                content=answer
            )
        )

        # Translate the answer into Indonesian if language choses is Indonesian
        if language == 'Indonesian':
            translated_answer = EasyGoogleTranslate(
                    source_language='en',
                    target_language='id'
                ).translate(
                    text=answer
                )

            streamlit.session_state.indonesian.append(
                AIMessage(
                    content=translated_answer
                )
            )

        # Stream each answer coming from Assistant
        output_stream = markdown_text[language][-1].content.splitlines()
        streamed_text = ''
        output_container = streamlit.chat_message(
            name='assistant'
        ).empty()

        for stream_line in output_stream:
            if stream_line != '':
                for stream in stream_line.split():
                    if stream in ['•']:
                        stream = '*'

                    time.sleep(0.1)
                    streamed_text += ' ' + stream
                    output_container.markdown(
                        body=streamed_text
                    )

            time.sleep(0.1)
            streamed_text += '\n'
            output_container.markdown(
                body=streamed_text
            )

if __name__ == '__main__':
    main()
