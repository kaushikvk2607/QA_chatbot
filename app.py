from flask import Flask, request, render_template
from flask_cors import CORS
from src.helper import download_hugging_face_embeddings
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from src.prompt import *
from langchain.vectorstores import FAISS
import os

app = Flask(__name__)
CORS(app)
qa = None

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def get_qa_chain():
    global qa

    if qa is None:
        print("Loading embeddings...")
        embeddings = download_hugging_face_embeddings()

        print("Loading FAISS index...")
        docsearch = FAISS.load_local(
            "faiss_index",
            embeddings
        )

        print("Loading LLM model...")
        llm = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q4_0 (1).bin",
            model_type="llama",
            config={
                "max_new_tokens": 150,
                "temperature": 0.2,
                "threads": os.cpu_count(),
            }
        )

        print("Setting up QA chain...")
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": PROMPT},
        )

        print("QA chain ready!")

    return qa


@app.route("/")
def home():
    return "Backend running..."


@app.route("/get", methods=["POST"])
def chat():
    try:
        data = request.get_json()

        if not data:
            return "No JSON received", 400

        msg = data.get("msg", "").strip()

        if not msg:
            return "Empty message", 400

        print(f"User query: {msg}")

        chain = get_qa_chain()
        result = chain({"query": msg})
        answer = result["result"]

        print(f"Answer: {answer}")

        return answer

    except Exception as e:
        print("Error:", str(e))
        return f"Error: {str(e)}", 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=False)