import pdfplumber
import os

# 현재 스크립트 기준 PDF 경로 지정
current_dir = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(current_dir, "KB 실버암 간편건강보험Plus.pdf")
txt_path = os.path.join(current_dir, "KB_실버암_간편건강보험Plus.txt")

# PDF 열어서 텍스트 추출 후 TXT로 저장
with pdfplumber.open(pdf_path) as pdf:
    with open(txt_path, 'w', encoding='utf-8') as f:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                f.write(text)
                f.write('\n')
s
print("✅ 텍스트 파일 저장 완료:", txt_path)
