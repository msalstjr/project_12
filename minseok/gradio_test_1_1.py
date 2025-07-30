import os
import shutil
import gradio as gr
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 경로 설정
VECTOR_ROOT = "vectorstores"
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(VECTOR_ROOT, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 전역 상태
uploaded_files = []
built_categories = set()

# 카테고리 키워드 기반 분류
category_keywords = {
    "cancer": ["암"],
    "medical": ["실손"],
    "accident": ["상해"],
    "fire": ["화재"]
}

def get_category_from_filename(filename):
    for category, keywords in category_keywords.items():
        if any(keyword in filename for keyword in keywords):
            return category
    return "cancer"  # 기본 fallback

# 텍스트 분할기 / 임베딩 모델
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
embedding = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

# PDF 업로드 처리
def upload_pdf(file):
    filename = os.path.basename(file.name)
    save_path = os.path.join(UPLOAD_DIR, filename)
    shutil.copy(file.name, save_path)
    uploaded_files.append(save_path)
    return f"✅ '{filename}' 업로드 완료. 더 업로드할 파일이 있나요?"

# 벡터스토어 생성
def build_vectorstores():
    global built_categories
    built_categories.clear()
    for path in uploaded_files:
        filename = os.path.basename(path)
        category = get_category_from_filename(filename)
        print(f"[+] Building vectorstore for: {filename} → category={category}")

        loader = PyPDFLoader(path)
        pages = loader.load_and_split()
        docs = text_splitter.split_documents(pages)

        vectorstore_path = os.path.join(VECTOR_ROOT, category)
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
        vectorstore = FAISS.from_documents(docs, embedding=embedding)
        vectorstore.save_local(vectorstore_path)
        built_categories.add(category)

    return f"📚 총 {len(built_categories)}개 카테고리 벡터스토어 생성 완료: {', '.join(built_categories)}"

# 질문 → 카테고리 추론
category_keywords_full = {
    "cancer": [
        "암", "유사암", "특정암", "전이암", "재진단암", "갑상선암", "폐암", "간암", "췌장암",
        "소화기관암", "혈액암", "생식기암", "항암", "항암치료", "방사선치료", "항암방사선",
        "항암약물", "표적항암", "호르몬약물", "CAR-T", "진단비", "암진단비", "암사망"
    ],
    "medical": [
        "실손", "의료비", "입원", "통원", "진료비", "검사비", "수술", "응급실", "치료",
        "자기부담금", "보험금 한도", "진단서", "의무기록", "보상종목", "다수보험", "연대책임"
    ],
    "accident": [
        "상해", "재해", "사고", "교통사고", "골절", "화상", "후유장해", "사고사망",
        "입원비", "수술비", "상해사망", "상해특약"
    ],
    "fire": [
        "화재", "폭발", "붕괴", "누수", "도난", "배상책임", "재산", "가재도구",
        "복구", "손해", "피해", "주택", "화재보험", "화재사고"
    ]
}

def classify_question(question):
    for category, keywords in category_keywords_full.items():
        if any(keyword in question for keyword in keywords):
            return category
    return "cancer"

# LLM & RAG 구성
llm = ChatAnthropic(model="claude-opus-4-20250514", temperature=0, max_tokens=1024, api_key=ANTHROPIC_API_KEY)
prompt = hub.pull("rlm/rag-prompt")
llm_only_chain = llm | StrOutputParser()

def answer_question(question):
    category = classify_question(question)
    vectorstore_path = os.path.join(VECTOR_ROOT, category)

    if not os.path.exists(vectorstore_path):
        return f"❌ 선택된 카테고리 '{category}'의 벡터스토어가 존재하지 않습니다. 먼저 관련 PDF를 업로드하고 벡터스토어를 생성하세요."

    vectorstore = FAISS.load_local(vectorstore_path, embeddings=embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()

    rag_chain = ( {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser() )

    llm_answer = llm_only_chain.invoke(question)
    rag_answer = rag_chain.invoke(question)

    result = f"[📂 선택된 카테고리: {category}]\n\n"
    result += f"💬 LLM 단독 응답:\n{llm_answer}\n\n"
    result += f"📚 RAG 기반 응답:\n{rag_answer}"
    return result

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📄 보험약관 RAG 시스템")

    with gr.Row():
        file_input = gr.File(label="PDF 업로드", file_types=[".pdf"], file_count="single")
        upload_btn = gr.Button("📥 PDF 추가")
    upload_output = gr.Textbox(label="업로드 상태")

    confirm_btn = gr.Button("✅ PDF 업로드 완료 & 벡터스토어 생성")
    confirm_output = gr.Textbox(label="생성 상태")

    question_input = gr.Textbox(label="질문 입력")
    question_btn = gr.Button("🔍 질문하기")
    answer_output = gr.Textbox(label="답변 출력", lines=10)

    upload_btn.click(upload_pdf, inputs=[file_input], outputs=[upload_output])
    confirm_btn.click(build_vectorstores, outputs=[confirm_output])
    question_btn.click(answer_question, inputs=[question_input], outputs=[answer_output])

demo.launch()

