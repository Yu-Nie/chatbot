from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import CSVLoader
from langchain_openai import OpenAI

load_dotenv('.env')

text_loader = CSVLoader('docs/OTL_edited.csv')
documents = text_loader.load()

chain = load_qa_chain(llm=OpenAI())
# query = 'suggest me 3 books about algebra'
query = 'who wrote college algebra'
response = chain.invoke({"input_documents": documents, "question": query})
print(response["output_text"])
