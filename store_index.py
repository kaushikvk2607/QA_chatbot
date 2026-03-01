from src.helper import load_pdf, text_split, download_hugging_face_embeddings
import os
from langchain.vectorstores import FAISS
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()
#


docsearch = FAISS.from_texts(
    [t.page_content for t in text_chunks],
    embeddings
)
docsearch.save_local("faiss_index")
