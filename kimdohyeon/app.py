import os
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print("OpenAI API Key:", OPENAI_API_KEY)  # 디버깅용 출력
# 벡터스토어 저장 함수
def load_and_store_pdf(file_path, persist_dir="vector_store"):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectordb = Chroma.from_documents(split_docs, embedding, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

# RAG 체인 생성
def get_rag_chain(persist_dir="vector_store"):
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )
    return chain

# LLM vs RAG 응답 비교
def compare_llm_vs_rag(query, rag_chain):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, openai_api_key=OPENAI_API_KEY)
    llm_response = llm.predict(query)
    rag_response = rag_chain.run(query)

    return llm_response, rag_response

# Gradio 인터페이스 함수
def process(pdf_file, question):
    # PDF 저장
    upload_path = "upload.pdf"
    with open(upload_path, "wb") as f:
        f.write(pdf_file)  # 수정된 부분

    # 벡터스토어 생성
    load_and_store_pdf(upload_path)

    # RAG 체인 생성
    rag = get_rag_chain()

    # 질문 처리
    llm_response, rag_response = compare_llm_vs_rag(question, rag)

    return f"🔹 LLM 응답:\n{llm_response}", f"🔹 RAG 응답:\n{rag_response}"


# Gradio UI 정의
with gr.Blocks() as demo:
    gr.Markdown("## 📄 PDF 기반 질의응답 (LLM vs RAG)")
    with gr.Row():
        pdf_input = gr.File(label="PDF 파일 업로드", type="binary")
        question_input = gr.Textbox(label="질문 입력", placeholder="예: 유사암진단비 청구 시 갑상선암 진단 기준은?")
    with gr.Row():
        btn = gr.Button("질문하기")
    with gr.Row():
        llm_output = gr.Textbox(label="LLM 응답 (GPT-3.5)", lines=10)
        rag_output = gr.Textbox(label="RAG 응답 (문서 기반)", lines=10)

    btn.click(fn=process, inputs=[pdf_input, question_input], outputs=[llm_output, rag_output])

# 실행
if __name__ == "__main__":
    demo.launch()
