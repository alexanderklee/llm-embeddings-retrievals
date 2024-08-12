from langchain.embeddings.base import Embeddings
from langchain.vectorstores.chroma import Chroma
from langchain.schema import BaseRetriever

class RedundantFilterRetriever(BaseRetriever):
    # require caller to include a method to generate embeddings
    embeddings: Embeddings
    # require caller to include a database object
    chroma: Chroma
    
    def get_relevant_documents(self,query):
        # calculate embeddings for query string
        emb = self.embeddings.embed_query(query)
        
        # take embeddings and feed them into the
        # max_marginal_relevance_search_by_vector function
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
    
            # ensure database only returns dis-similar documents
            lambda_mult=0.8 # Values: 0-1 where higher value is "more similar"
        )
        return []
    async def aget_relevant_documents(self):
        return []