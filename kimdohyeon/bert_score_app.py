import os
import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from bert_score import score

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

from anthropic import Anthropic

# 🔹 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 🔹 Claude 메시지 기반 응답 함수
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)

def claude_chat_message(prompt, model="claude-3-opus-20240229"):
    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text

# 🔹 벡터스토어 생성
def load_and_store_pdf(file_path, persist_dir="vector_store"):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    split_docs = splitter.split_documents(docs)
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectordb = Chroma.from_documents(split_docs, embedding, persist_directory=persist_dir)
    vectordb.persist()
    return vectordb

# 🔹 RAG 체인 생성
def get_rag_chain(model_name="gpt-3.5-turbo"):
    embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    vectordb = Chroma(persist_directory="vector_store", embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    llm = ChatOpenAI(model_name=model_name, temperature=0.2, openai_api_key=OPENAI_API_KEY)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return chain

# 🔹 모델별 LLM + RAG 응답 생성
def compare_all_models(question):
    results = {}

    # 🔸 GPT-3.5
    llm_35 = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
    results['gpt-3.5-LLM'] = llm_35.predict(question)
    results['gpt-3.5-RAG'] = get_rag_chain("gpt-3.5-turbo").run(question)

    # 🔸 GPT-4
    llm_4 = ChatOpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)
    results['gpt-4-LLM'] = llm_4.predict(question)
    results['gpt-4-RAG'] = get_rag_chain("gpt-4").run(question)

    # 🔸 Claude 3 Opus
    results['claude-3-LLM'] = claude_chat_message(question, "claude-3-opus-20240229")
    rag_query = get_rag_chain("gpt-3.5-turbo").run(question)
    results['claude-3-RAG'] = claude_chat_message(rag_query, "claude-3-opus-20240229")

    # 🔸 Claude 4 Opus
    results['claude-4-LLM'] = claude_chat_message(question, "claude-opus-4-20250514")
    results['claude-4-RAG'] = claude_chat_message(rag_query, "claude-opus-4-20250514")

    return results

# 🔹 BERTScore 시각화
def plot_bert_score(responses, reference):
    model_names = list(responses.keys())
    cand_list = list(responses.values())
    P, R, F1 = score(cand_list, [reference]*len(cand_list), lang='en', verbose=False)
    df = pd.DataFrame({
        "Model": model_names,
        "BERTScore": F1.numpy()
    }).sort_values("BERTScore", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(df["Model"], df["BERTScore"])
    plt.xticks(rotation=45)
    plt.title("BERTScore (F1) for LLM 응답")
    plt.ylim(0, 1)
    plt.tight_layout()
    path = "bert_score.png"
    plt.savefig(path)
    return path

# 🔹 Gradio용 메인 처리 함수
def process(pdf_file, question):
    with open("upload.pdf", "wb") as f:
        f.write(pdf_file)

    load_and_store_pdf("upload.pdf")

    responses = compare_all_models(question)

    # LLM 응답만 BERTScore 평가
    llm_only = {k: v for k, v in responses.items() if "LLM" in k}
    score_plot_path = plot_bert_score(llm_only, question)

    # 전체 응답 출력 정리
    output_text = "\n\n".join([f"🔹 {k}:\n{v}" for k, v in responses.items()])
    return output_text, score_plot_path

# 🔹 Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 PDF 기반 질의응답 시스템 (GPT & Claude + BERTScore 시각화)")
    with gr.Row():
        pdf_input = gr.File(label="PDF 업로드", type="binary")
        question_input = gr.Textbox(label="질문", placeholder="예: 암보험 면책기간은 몇 일인가요?")
    with gr.Row():
        btn = gr.Button("질문하기")
    with gr.Row():
        response_text = gr.Textbox(label="모델 응답 모음", lines=20)
    with gr.Row():
        image_output = gr.Image(label="BERTScore 그래프")

    btn.click(fn=process, inputs=[pdf_input, question_input], outputs=[response_text, image_output])

# 🔹 실행
if __name__ == "__main__":
    demo.launch()
