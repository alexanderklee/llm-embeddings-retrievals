from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from langchain.vectorstores.chroma import Chroma

import signal
import sys
from dotenv import load_dotenv

# Note: EmbeddingsRedundantFilter has no way of accessing or sharing
#       embeddings from the RetrievalQA chain. Will need to make a
#       custom retriver that implements get_relevant_documents. This
#       will remove duplicate records sent and avoid over-using the
#       embeddings endpoint

def signal_handler(sig, frame):
    print('You pressed Ctrl+C, goodbye!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

load_dotenv()

embeddings = OpenAIEmbeddings()

# Quick test to validate embedding call
# emb = embeddings.embed_query("hello world")
# print(emb)
# sys.exit(0)

# split text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200, # 200 chars long max
    chunk_overlap=0 # no overlap between chunks
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(
    text_splitter=text_splitter
)

# Create new chroma db called emb and generate
# embeddings from openAI
# Note: this will cost $$ and avoid re-running until retrieval chain is done
#       to avoid dupilcate chunks/documents in chroma
db = Chroma.from_documents(
    docs,
    embedding=embeddings, # note: embedding "minus" the "s"
    persist_directory="emb"
)


# Generate embeddings for chunked text


results = db.similarity_search_with_score(
    "What is an interesting fact about the English language?",
    k=2 # returns the # of the semantic search results
    )
for result in results:
    print("\n")
    print (result[1]) # simiilarity score
    print(result[0].page_content) # actual documents / chunks

# Need to see chunks for debugging
# for doc in docs:
#    print(doc.page_content)
#    print("\n")

# print (docs)
