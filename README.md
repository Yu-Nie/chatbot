# Chatbot

Welcome to the Chatbot project! This chatbot is designed to read custmized information and provide seamless and interactive communication between users and automated systems. Utilizing advanced natural language processing techniques, the chatbot can understand and respond to user queries in a conversational manner.

There are lots of tutorials and online resources available, however some are outdated and some don't working at all.

I gathered the articles and some top rated repos about buidling a custmized chatbot, with ability to answer based on provided materials.

Here is a summary for the links and I also uploaded code that works perfectly fine itself or with some adjustments.

| Link | Last Updated Time | Purpose | Technology | Issues |
| ---- | ----------------- | ------- | ---------- | ------ |
| [LLama3-ChatPDF](https://github.com/Sh9hid/LLama3-ChatPDF) | 06/24 | pdf | LLama3, Langchain, Ollama, Streamlit | need to ask relevant questions right after a file is uploaded, otherwise this file won’t be recognized |
| [groq-llama3](https://github.com/eersnington/groq-llama3-pdf-rag) | 07/24 | pdf | Streamlit, Groq AI, Streamlit-Option-Menu, Langchain, FAIS | Some questions would be recognized as not related |
| [rag-llama3](https://lightning.ai/lightning-ai/studios/rag-using-llama-3-by-meta-ai?section=featured) | 07/24 | llama3 (llama3:8b-instruct-q4_1), hugging face | Can only process one file, no memory of previous files |
| [rag-langchain](https://python.langchain.com/v0.2/docs/tutorials/rag/) | - | web page | Langchain,  openAI (gpt-4o-mini) | - | 
| [chat-with-your-city](https://www.e2enetworks.com/blog/chat-with-your-city-steps-to-build-an-ai-chatbot-using-llama-3-and-dspy) | 06/24 | pdf | Llama3, DSPy | failed to run: Your session crashed after using all available RAM |
| [llamaindex-chatbot](https://blog.streamlit.io/build-a-chatbot-with-custom-data-sources-powered-by-llamaindex/) | 08/23 | csv, pdf | LlamaIndex, OpenAI (gpt-3.5-turbo), streamlit, nltk | - |	
| [openwebui-llama3](https://usamakhaninsights.medium.com/how-to-build-your-own-custom-chatbot-with-llama-3-1-and-openwebui-a-step-by-step-guide-66566876b121) | 09/24 | LLaMa3, Ollama, WebUI | didn’t mention about using custom data |
| [tabular-data-rag](https://medium.com/intel-tech/tabular-data-rag-llms-improve-results-through-data-table-prompting-bcb42678914b) | 05/24 | tabular data | OpenAI (gpt-4), pandas | • it only processes one page that has table • Throw UnboundLocalError: local variable 'table_df' referenced before assignment if no table on the page" |
| [text-to-pandas](https://www.youtube.com/watch?app=desktop&v=L1o1VPVfbb0) | 02/24 | - | - | "https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_pandas/ |
| [text-to-sql](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/") | - | - | llama-index-llms-openai (gpt-3.5-turbo), llama-index-experimenta | - |
| [langchain-ollama](https://medium.aiplanet.com/implementing-rag-using-langchain-ollama-and-chainlit-on-windows-using-wsl-92d14472f15d) | 11/23 | pdf | Ollama, Chainlit(Langchain) | fail to upload files |
| [chainlit-langchain](https://medium.com/@cleancoder/build-a-chatbot-in-minutes-with-chainlit-gpt-4-and-langchain-7690968578f0) | 11/23 | - | AzureChatOpenAI, Chainlit, Langchain | unclear about model name and API key, unable to connect |
| [chainlit-chatbot](https://tinztwinshub.com/software-engineering/build-a-local-chatbot-in-minutes-with-chainlit/) | 06/24 | - | Ollama (llama3.2), Chainlit	| fail to upload files |
| [multi-doc-reader](https://betterprogramming.pub/building-a-multi-document-reader-and-chatbot-with-langchain-and-chatgpt-d1864d47e339) | 05/23 | multi-doc reader, pdf, docx, txt	| OpenAI | https://github.com/smaameri/multi-doc-chatbot/tree/master |can’t process large pdfs |
| [streamlit-rag](https://www.bluebash.co/blog/pdf-csv-chatbot-rag-langchain-streamlit/) | 11/24 | pdf, csv | OpenAI, Langchain, streamlit | - |
| [rag-openai](https://medium.com/thedeephub/rag-chatbot-powered-by-langchain-openai-google-generative-ai-and-hugging-face-apis-6a9b9d7d59db) | 2/24 | txt, pdf, CSV, docx | Langchain, OpenAI, Google Generative AI, Hugging Face APIs, Streamlit | an incompatible version of langchain or pydantic ; tried some methods to match those versions but still failed to run |
                    