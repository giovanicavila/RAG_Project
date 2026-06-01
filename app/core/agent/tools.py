from app.core.retrieval.retriever import retriever


class RetrievalTool:
    def run(self, query: str, top_k=None):
        return retriever.search(query, top_k)


retrieval_tool = RetrievalTool()
