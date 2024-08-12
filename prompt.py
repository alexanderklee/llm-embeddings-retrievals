import langchain.document_loaders
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# custom retriever
from redundant_filter_retriever import RedundantFilterRetriever

import langchain
langchain.debug=True

import signal
import sys
from dotenv import load_dotenv

def signal_handler(sig, frame):
    print('You pressed Ctrl+C, goodbye!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()

db = Chroma(
    persist_directory="emb",
    embedding_function=embeddings
)

# No need to use Chroma retriever here, use custom one instead 
# retriever = db.as_retriever()
retriever = RedundantFilterRetriever(
    embeddings=embeddings,
    chroma=db
)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff" # Other options: Refine, MapReduce, MapReRank
)                      # "Stuff" means just stuff the retreived data into the SystemMessage

result = chain.run("What is an interesting fact about the English language?")

print (result)