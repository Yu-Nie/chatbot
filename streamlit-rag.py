import os
import pathlib
import pdb
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import streamlit as st
from pypdf import PdfReader
from tempfile import NamedTemporaryFile
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader

# https://www.bluebash.co/blog/pdf-csv-chatbot-rag-langchain-streamlit/

def convert_to_json(document_content):
    messages = [
        SystemMessage(content="System message"),
        HumanMessage(content=document_content)
    ]
    answer = chat.invoke(messages)
    return answer.content

def prepare_files(files):
    document_content = ""
    for file_name in os.listdir(files):
        file = os.path.join(folder_path, file_name)
        # if file.type == 'application/pdf':
        if file_name.endswith('.pdf'):
            # document_content += handle_pdf_file(file)
            document_content += handle_pdf_with_images(file)
        # elif file.type == 'text/csv':
        elif file_name.endswith('.csv'):
            document_content += handle_csv_file(file)
        else:
            st.write('File type is not supported!')
        # document_content += "".join(page_contents)
    return document_content

def handle_pdf_file(pdf_file):
    # document_content = ''
    # with pdf_file as file:
    #     pdf_reader = PdfReader(file)
    #     page_contents = []
    #     for page in pdf_reader.pages:
    #         page_contents.append(page.extract_text())
    #     document_content += "\n".join(page_contents)
    # return document_content
    pdf_reader = PdfReader(pdf_file)
    page_contents = [page.extract_text() for page in pdf_reader.pages]
    return "\n".join(page_contents)


def handle_pdf_with_images(pdf_file):
    text_content = ""
    pdf_reader = fitz.open(pdf_file)

    for page_num in range(len(pdf_reader)):
        page = pdf_reader[page_num]

        # Extract text from the page
        text_content += page.get_text()

        # Extract images from the page
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf_reader.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Optional: Preprocess the image for better OCR
            gray_image = image.convert("L")

            # Perform OCR on the image
            image_text = pytesseract.image_to_string(gray_image)
            text_content += f"\n[Image Text {page_num}-{img_index}]: {image_text}"

    pdf_reader.close()
    return text_content


def handle_csv_file(csv_file):
    with csv_file as file:
        uploaded_file = file.read()
        with NamedTemporaryFile(dir='.', suffix='.csv') as f:
            f.write(uploaded_file)
            f.flush()
            loader = CSVLoader(file_path=f.name)
            document_content = "".join([doc.page_content for doc in loader.load()])
    return document_content

st.set_page_config(page_title='AI PDF Chatbot', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("PDF Chatbot")

# files = st.file_uploader("Upload PDF and CSV files:", accept_multiple_files=True, type=["csv", "pdf"])
folder_path = './data'


openai_key = "" # put your openAI API key

if openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    chat = ChatOpenAI(model_name='gpt-4', temperature=0)
    embeddings = OpenAIEmbeddings()

query = st.text_input("Enter your query for the document data:")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)

if st.button("Get Answer to Query"):
    if folder_path and openai_key and query:
        document_content = prepare_files(folder_path)
        chunks = text_splitter.split_text(document_content)
        db = FAISS.from_texts(chunks, embeddings)
        chain = load_qa_chain(chat, chain_type="stuff", verbose=True)
        docs = db.similarity_search(query)
        response = chain.run(input_documents=docs, question=query)
        st.write("Query Answer:")
        st.write(response)
    else:
        st.warning("Please upload PDF and CSV files, enter your OpenAI API key, and enter your query")