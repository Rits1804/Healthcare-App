import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
from langchain.prompts import PromptTemplate
import os
from audio_handler import transcribe_audio

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to get Gemini response for image input
def get_gemini_response(input, image, prompt):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content([input, image[0], prompt])
    return response.text

# Function to handle user input from audio
def user_input_audio(user_question, transcribed_text, prompt):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([user_question, transcribed_text, prompt])
    return response.text

# Function to set up image input
def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

# Function to create conversational chain for PDF documents
def get_conversational_chain(prompt):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt_template = """
        as it is medical supervsion who analyze or give ans on user's question Please provide a detailed response based on the information from the uploaded document.
        Ensure that you include all relevant details. When asked about related topics not directly 
        covered in the document, generate a response based on its content and its relevance to the query. 
        If the user requests a summary of the report, highlight the key points included in the document 
        rather than providing a detailed response. Even if they ask something else, try to provide a 
        detailed answer based on the contents of the document. Your answer should be sufficiently detailed,
        especially for large documents, to ensure satisfaction. 

Context:
{context}
Question:
{question}
Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input for PDF documents
def user_input(user_question, pdf_docs, prompt):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(prompt)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Healthcare App")
    st.header("**Welcome to the Healthcare App**")
    st.markdown("**Ask any query related to the provided document**")
        
    # Sidebar for PDF related interactions
    st.sidebar.header("PDF Related")
    uploaded_file_pdf = st.sidebar.file_uploader("upload your medical report (PDF)")
    user_question_pdf = st.sidebar.text_input("Ask a Question from the report file")
    pdf_button = st.sidebar.button("Get PDF Answer")
     
    # Sidebar for image related interactions
    st.sidebar.header("Image Related")
    uploaded_file_image = st.sidebar.file_uploader("upload your medical report(Images)")
    user_question_image = st.sidebar.text_input("Ask a Question about the Image")
    image_button = st.sidebar.button("Get Image Answer")

    # Sidebar for audio related interactions
    st.sidebar.header("Audio Related")
    uploaded_file_audio = st.sidebar.file_uploader("Upload an audio file", type=["mp3", "wav"])
    user_question_audio = st.sidebar.text_input("Ask a Question about the Audio")
    audio_button = st.sidebar.button("Get Audio Answer")

    # Handle PDF input
    if pdf_button:
        if uploaded_file_pdf is not None and user_question_pdf:
            prompt = user_question_pdf
            response_pdf = user_input(user_question_pdf, [uploaded_file_pdf], prompt)
            st.write("Reply (PDF): ")
            st.write(response_pdf)
    
    # Handle image input
    if image_button:
        if uploaded_file_image is not None and user_question_image:
            prompt = user_question_image
            image = Image.open(uploaded_file_image)
            st.image(image, caption="Uploaded Image.", use_column_width=True)
            
            if user_question_image:
                image_data = input_image_setup(uploaded_file_image)
                response_image = get_gemini_response(user_question_image, image_data, prompt)
                st.subheader("The Response (Image) is")
                st.write(response_image)
    
    # Handle audio input
    if audio_button:
        if uploaded_file_audio is not None and user_question_audio:
            # Transcribe audio
            audio_bytes = uploaded_file_audio.read()
            transcribed_text = transcribe_audio(audio_bytes)  # Assigning a value to transcribed_text
            prompt = user_question_audio
            # Process user's question and provide answer based on the transcribed text
            answer_audio = user_input_audio(user_question_audio, transcribed_text, prompt)
            st.write("Answer: ")
            st.write(answer_audio)

if __name__ == "__main__":
    main()
