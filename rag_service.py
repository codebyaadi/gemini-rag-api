import os
import google.generativeai as genai
from chromadb import Client as ChromaClient


class RAGService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)

        self.generative_model = genai.GenerativeModel("gemini-1.5-flash-002")
        self.embedding_model_name = "models/embedding-001"

        self.chroma_client = ChromaClient()
        self.collection = self.chroma_client.create_collection(name="knowledge_base")

        self._load_and_embed_knowledge_base()

    def _load_and_embed_knowledge_base(self):
        """Loads the knowledge base from a file and stores embeddings."""
        try:
            with open("knowledge_base.txt", "r") as f:
                documents = [line.strip() for line in f.readlines() if line.strip()]

            print(f"Embedding {len(documents)} document(s)...")
            response = genai.embed_content(
                model=self.embedding_model_name,
                content=documents,
                task_type="retrieval_document",
            )
            embeddings = response["embedding"]

            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                ids=[f"id_{i}" for i in range(len(documents))],
            )
            print("Knowledge base loaded and embedded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the knowledge base: {e}")

    def answer_question(self, question: str) -> str:
        """Answers a question using the RAG pattern."""
        question_embedding_response = genai.embed_content(
            model=self.embedding_model_name,
            content=question,
            task_type="retrieval_query",
        )
        question_embedding = question_embedding_response["embedding"]

        results = self.collection.query(
            query_embeddings=[question_embedding], n_results=2
        )
        context = "\n".join(results["documents"][0])

        prompt = f"""
        Using ONLY the following context, answer the question.
        If the context does not contain the answer, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """

        try:
            response = self.generative_model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            return "Sorry, I encountered an error while generating the answer."
