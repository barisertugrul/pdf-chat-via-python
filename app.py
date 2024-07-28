from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

APP_ICON = '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" id="bubble"><path d="M6 45.414V36H3c-1.654 0-3-1.346-3-3V7c0-1.654 1.346-3 3-3h42c1.654 0 3 1.346 3 3v26c0 1.654-1.346 3-3 3H15.414L6 45.414zM3 6a1 1 0 0 0-1 1v26a1 1 0 0 0 1 1h5v6.586L14.586 34H45a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1H3z"></path><circle cx="16" cy="20" r="2"></circle><circle cx="32" cy="20" r="2"></circle><circle cx="24" cy="20" r="2"></circle></svg>'
st.set_page_config(page_title="Ask To PDF App", page_icon=APP_ICON)

svg_icon = """
<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" id="bubble" fill="white">
  <path d="M6 45.414V36H3c-1.654 0-3-1.346-3-3V7c0-1.654 1.346-3 3-3h42c1.654 0 3 1.346 3 3v26c0 1.654-1.346 3-3 3H15.414L6 45.414zM3 6a1 1 0 0 0-1 1v26a1 1 0 0 0 1 1h5v6.586L14.586 34H45a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1H3z"></path>
  <circle cx="16" cy="20" r="2"></circle>
  <circle cx="32" cy="20" r="2"></circle>
  <circle cx="24" cy="20" r="2"></circle>
</svg>
"""

def main():
    load_dotenv()

    # Sayfa Başlığı
    st.markdown(f'<span style="font-size: 30px, font-weight:600">Ask To</span> :blue[PDF] <span style="margin-left:10px"> {svg_icon}</span>', unsafe_allow_html=True)

    # İkon Atıfı
    st.markdown('<span style="font-size: 10px">The icon <a href="https://iconscout.com/icons/bubble" class="text-underline font-size-sm" target="_blank">Bubble</a> by <a href="https://iconscout.com/contributors/vincent-le-moign" class="text-underline font-size-sm" target="_blank">Vincent Le moign</a></span>', unsafe_allow_html=True)

    # API Anahtarı Formu
    api_key = st.text_input("Enter your OpenAI API Key", type="password")
    if not api_key:
        st.warning("Please enter your OpenAI API Key")
        return

    # Dosya Yükleyici
    pdf = st.file_uploader("Upload Your PDF File Here", type=["pdf"])

    # PDF Okuma
    text=""
    if pdf:
        reader = PdfReader(pdf)
        st.write(f"Number of pages in PDF: {len(reader.pages)}")

        for page in reader.pages:
            text += page.extract_text()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        try:
            chunks = text_splitter.split_text(text)
            # Embeddings
            embeddings = OpenAIEmbeddings(api_key=api_key)
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            # IndexError hatasını oluşturan kod
        except IndexError as e:
            # IndexError hatasını ele al, örneğin bir hata mesajı yazdır veya istisnayı logla
            print("IndexError:", e)


        # Kullanıcı Sorusu
        st.write("Please ask questions about PDF content")
        user_question = st.text_area("Enter Question Here")
        if user_question:
            st.write("Searching for answer...")
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(api_key=api_key)
            qa_chain = load_qa_chain(llm, chain_type="stuff")
            response = qa_chain.run(question=user_question, input_documents=docs)
            st.write(response)

if __name__ == "__main__":
    main()