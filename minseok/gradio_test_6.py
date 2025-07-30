import os
import shutil
import gradio as gr
from dotenv import load_dotenv

import fitz  # PyMuPDF
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

# 1. 환경설정 및 폴더 준비
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

VECTOR_ROOT = "vectorstores"
UPLOAD_DIR = "uploaded_pdfs"
TXT_DIR = "converted_txts"
os.makedirs(VECTOR_ROOT, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TXT_DIR, exist_ok=True)

uploaded_files = []
built_categories = set()

# 2. 카테고리 정의
category_keywords = {
    "cancer": ["암"],
    "medical": ["실손"],
    "accident": ["상해"],
    "fire": ["화재"],
}

def get_category_from_filename(filename):
    for category, keywords in category_keywords.items():
        if any(keyword in filename for keyword in keywords):
            return category
    return "cancer"  # fallback

# 3. PDF → 좌/우 텍스트 분리 함수 (cancer 전용)
def extract_left_right_text(pdf_path):
    doc = fitz.open(pdf_path)
    left_text_total = ""
    right_text_total = ""
    for page_num, page in enumerate(doc):
        width = page.rect.width
        blocks = page.get_text("blocks")
        left_text = ""
        right_text = ""
        for b in blocks:
            x0, y0, x1, y1, text, *_ = b
            center_x = (x0 + x1) / 2
            if center_x < width / 2:
                left_text += text.strip() + " "
            else:
                right_text += text.strip() + " "
        left_text_total += f"[페이지 {page_num + 1} - 좌]\n{left_text.strip()}\n\n"
        right_text_total += f"[페이지 {page_num + 1} - 우]\n{right_text.strip()}\n\n"
    return left_text_total + right_text_total

# 4. PDF → 전체 텍스트 추출 함수 (나머지 카테고리)
def extract_full_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
        full_text += "\n\n"
    return full_text

# 5. PDF 업로드 시 TXT 자동 변환 (카테고리별 분기)
def upload_pdf(file, file_list):
    filename = os.path.basename(file.name)
    save_path = os.path.join(UPLOAD_DIR, filename)
    txt_path = os.path.join(TXT_DIR, filename.replace(".pdf", ".txt"))

    shutil.copy(file.name, save_path)
    category = get_category_from_filename(filename)
    # cancer만 좌/우 분리, 나머지는 전체 텍스트
    if not os.path.exists(txt_path):
        if category == "cancer":
            txt = extract_left_right_text(save_path)
        else:
            txt = extract_full_text(save_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt)
    if save_path not in file_list:
        file_list.append(save_path)
    all_filenames = [os.path.basename(f) for f in file_list]
    return file_list, "\n".join(all_filenames)

# 6. 청크 및 임베딩
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=OPENAI_API_KEY,
)

# 7. 카테고리별 벡터스토어 생성
def build_vectorstores(file_list):
    global built_categories
    built_categories.clear()
    for path in file_list:
        filename = os.path.basename(path)
        category = get_category_from_filename(filename)
        print(f"[+] Building vectorstore for: {filename} → category={category}")
        txt_path = os.path.join(TXT_DIR, filename.replace(".pdf", ".txt"))
        loader = TextLoader(txt_path, encoding="utf-8")
        docs_raw = loader.load()
        docs = text_splitter.split_documents(docs_raw)
        vectorstore_path = os.path.join(VECTOR_ROOT, category)
        if os.path.exists(vectorstore_path):
            shutil.rmtree(vectorstore_path)
        vectorstore = FAISS.from_documents(docs, embedding=embedding)
        vectorstore.save_local(vectorstore_path)
        built_categories.add(category)
    return f"📚 총 {len(built_categories)}개 카테고리 벡터스토어 생성 완료: {', '.join(built_categories)}"

# 8. 카테고리 분류 (질문 기준)
category_keywords_full = {
    "cancer": [
        "암", "유사암", "특정암", "전이암", "재진단암", "갑상선암", "폐암", "간암", "췌장암",
        "소화기관암", "혈액암", "생식기암", "항암", "항암치료", "방사선치료", "항암방사선",
        "항암약물", "표적항암", "호르몬약물", "CAR-T", "진단비", "암진단비", "암사망"
    ],
    "medical": [
        "실손", "실비", "실손의료비보험"
    ],
    "accident": [
        "상해", "재해", "교통사고", "후유장해", "사고사망",
        "상해사망", "상해특약", "안전벨트", "마라톤"
    ],
    "fire": [
        "화재", "폭발", "붕괴", "누수", "도난", "배상책임", "재산", "가재도구",
        "복구", "손해", "피해", "주택", "화재보험", "화재사고", "유리손해"
    ]
}

def classify_question(question):
    for category, keywords in category_keywords_full.items():
        if any(keyword in question for keyword in keywords):
            return category
    return "cancer"

# 9. LLM 및 프롬프트 세팅
llm = ChatAnthropic(
    model="claude-opus-4-20250514",
    temperature=0,
    max_tokens=1024,
    api_key=ANTHROPIC_API_KEY,
)
prompt = hub.pull("rlm/rag-prompt")
llm_only_chain = llm | StrOutputParser()

# 10. 질문 답변 함수 (RAG)
def answer_question(question, chat_history):
    category = classify_question(question)
    vectorstore_path = os.path.join(VECTOR_ROOT, category)
    if not os.path.exists(vectorstore_path):
        msg = f"❌ 선택된 카테고리 '{category}'의 벡터스토어가 존재하지 않습니다. 먼저 관련 PDF를 업로드하고 벡터스토어를 생성하세요."
        chat_history.append((question, msg))
        return chat_history
    vectorstore = FAISS.load_local(vectorstore_path, embeddings=embedding, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} |
        prompt | llm | StrOutputParser()
    )
    rag_answer = rag_chain.invoke(question)
    kor_category = {
        "cancer": "암 보험",
        "medical": "실비 보험",
        "accident": "상해 보험",
        "fire": "화재보험"
    }.get(category, category)
    combined = f"[📂 선택된 카테고리: {kor_category}]\n\n"
    combined += f"📚 RAG 기반 응답:\n{rag_answer}"
    chat_history.append((question, combined))
    return chat_history

# 11. Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📄 보험약관 RAG 시스템")
    with gr.Row():
        file_input = gr.File(label="PDF 업로드", file_types=[".pdf"], file_count="single")
        upload_btn = gr.Button("📥 PDF 추가")
    upload_status = gr.Textbox(label="업로드된 PDF 목록", interactive=False)
    state_files = gr.State([])
    state_chat = gr.State([])
    confirm_btn = gr.Button("✅ PDF 업로드 완료 & 벡터스토어 생성")
    confirm_output = gr.Textbox(label="생성 상태", interactive=False)
    gr.Markdown("### 💬 보험 약관 챗봇 (Chat RAG)")
    with gr.Row():
        chatbot = gr.Chatbot(label="챗봇 대화 내역", height=400, show_copy_button=True, avatar_images=(None, None))
    chat_input = gr.Textbox(label="질문을 입력하세요", placeholder="궁금한 점을 입력 후 Enter", lines=1)
    upload_btn.click(
        upload_pdf,
        inputs=[file_input, state_files],
        outputs=[state_files, upload_status]
    )
    confirm_btn.click(
        build_vectorstores,
        inputs=[state_files],
        outputs=[confirm_output]
    )
    chat_input.submit(
        answer_question,
        inputs=[chat_input, state_chat],
        outputs=[chatbot],
        queue=True
    ).then(
        lambda chat_input, chat_history: ("", chat_history),
        inputs=[chat_input, state_chat],
        outputs=[chat_input, state_chat]
    )
demo.queue()
demo.launch()
