from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from datasets import Dataset
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

TEST_QUERIES = [
    {"question": "What is ESRS S1 about?", "ground_truth": "ESRS S1 covers disclosures related to the own workforce."},
    {"question": "What is the purpose of Appendix A in ESRS S1?", "ground_truth": "Appendix A provides application requirements for ESRS S1."},
    {"question": "What are the disclosure requirements related to ESRS 2 SBM-3?", "ground_truth": "ESRS S1 requires disclosure of material impacts and risks related to own workforce."},
]

embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=os.getenv("MISTRAL_API_KEY"))
vectorstore = Chroma(persist_directory=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"), embedding_function=embeddings, collection_name=os.getenv("COLLECTION_NAME", "esg_documents"))
llm = ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0)

questions, answers, contexts, ground_truths = [], [], [], []
for item in TEST_QUERIES:
    q = item["question"]
    docs = vectorstore.similarity_search(q, k=4)
    ctx = [doc.page_content for doc in docs]
    response = llm.invoke(f"Answer based only on:\n{chr(10).join(ctx)}\n\nQuestion: {q}\nAnswer:")
    questions.append(q)
    answers.append(response.content)
    contexts.append(ctx)
    ground_truths.append(item["ground_truth"])
    print(f"✅ {q[:60]}...")

dataset = Dataset.from_dict({"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths})

print("\n🔍 Running RAGAS evaluation...")
result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_precision], llm=llm, embeddings=embeddings)

print("\n📊 RAGAS Results:")
print(result)
