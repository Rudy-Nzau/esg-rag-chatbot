"""
CrossEncoder Reranker
Reclasse les chunks récupérés par pertinence réelle
"""

from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", top_k: int = 4):
        self.model = CrossEncoder(model_name)
        self.top_k = top_k
        print(f"✅ CrossEncoder reranker loaded: {model_name}")

    def rerank(self, query: str, documents: list) -> list:
        """
        Prend N documents, les reclasse par score de pertinence,
        retourne les top_k meilleurs.
        """
        if not documents:
            return documents

        # Paires (query, chunk) pour le CrossEncoder
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        # Trier par score décroissant
        scored_docs = sorted(
            zip(scores, documents),
            key=lambda x: x[0],
            reverse=True
        )

        top_docs = [doc for _, doc in scored_docs[:self.top_k]]
        logger.info(f"Reranked {len(documents)} → kept top {len(top_docs)}")
        return top_docs
