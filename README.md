# Healthcare-App
This Streamlit application allows users to upload medical reports in various formats (PDF, images, audio) and get detailed responses to their queries using the Google Gemini Pro API. and OpenAI Whisper.

## Features
PDF Support: Upload medical reports in PDF format and ask questions to get detailed responses.

Image Support: Upload medical reports as images and ask questions related to the images.

Audio Support: Upload audio files containing medical information, transcribe the audio, and ask questions to get responses based on the transcriptions


## Installation

1.Clone the repository

2.Install dependencies

3.Set up environment variables(GOOGLE_API_KEY) https://aistudio.google.com/app/apikey

## Dependencies
streamlit: Web application framework.


PyPDF2: Library for extracting text from PDF files.

langchain: Library for handling text processing and embeddings.

google-generativeai: Library to interact with Google Gemini Pro API.

PIL: Python Imaging Library for handling images.

python-dotenv: Load environment variables from a .env file.

audio_handler: Custom module for handling audio transcriptions.

## Usage
streamlit run app.py
