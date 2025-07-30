# resume_agent_app.py

import re
import os
from dotenv import load_dotenv
from pathlib import Path
import streamlit as st
import openai
import streamlit as st



from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
openai.api_key = st.secrets["OPENAI_API_KEY"]

# 📎 Manual case study URL map
case_study_links = {
    "Demonstrating Strategic P&L Ownership and Customer.md": "https://www.canva.com/design/DAGncSFIvHg/CK8h5GKzBCrt2ig2gS0bMw/view?utm_content=DAGncSFIvHg&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h7f7803d827",
    "Driving Adoption & Retention.md": "https://www.canva.com/design/DAGnjKSGHNk/4T3MK6YELPATzxtgV_Vd-Q/view?utm_content=DAGnjKSGHNk&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hdaf63b95d2",
    "Leading Culture-Driven Transformation.md": "https://www.canva.com/design/DAGoGUoywCQ/hG7e74REdSTCz1AHBTWWnQ/view?utm_content=DAGoGUoywCQ&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h8222948442",
    "Margin Improvement & Revenue Integrity.md": "https://www.canva.com/design/DAGndX1mdjw/G5xk6W4NRTNT-SFe76aVRQ/view?utm_content=DAGndX1mdjw&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h779a5f4e4a",
    "Scaling Customer Experience through VOC.md": "https://www.canva.com/design/DAGndukdNyE/kcFNsRRrvVbuzJkrEs5sdQ/view?utm_content=DAGndukdNyE&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h39a6f25d57",
    "automation.md": "https://www.canva.com/design/DAGndukdNyE/kcFNsRRrvVbuzJkrEs5sdQ/view?utm_content=DAGndukdNyE&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h39a6f25d57"
}

# 📄 Load documents from /docs
@st.cache_data
def load_documents(folder_path="docs"):
    import re  # Make sure this is imported at the top
    docs = []
    folder = Path(folder_path)
    
    for file_path in folder.glob("*"):
        ext = file_path.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif ext == ".docx":
            loader = Docx2txtLoader(str(file_path))
        elif ext in [".txt", ".md"]:
            loader = TextLoader(str(file_path))
        else:
            print(f"Skipping unsupported file: {file_path.name}")
            continue

        raw_docs = loader.load()
        file_name = file_path.name
        source_url = case_study_links.get(file_name, "")

        # Clean title: remove suffix starting with first digit, keep readable name
        clean_title = re.sub(r"\s?\d.*", "", file_name).replace("_", " ").replace(file_path.suffix, "").strip()

        # Manual override for automation case
        if file_name == "automation.md":
            clean_title = "Driving Automation and Efficiency"

        # Attach metadata
        for doc in raw_docs:
            doc.metadata["source_url"] = source_url
            doc.metadata["title"] = clean_title

        docs.extend(raw_docs)

    return docs


# 🧠 Set up persistent memory
if "memory" not in st.session_state:
    from langchain.memory import ConversationBufferMemory
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer",
    )

#Initialize Chain
@st.cache_resource
def initialize_chain():
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    embedding = OpenAIEmbeddings(openai_api_key=openai_key)
    vectordb = FAISS.from_documents(split_docs, embedding)

    retriever = vectordb.as_retriever(search_kwargs={'k':10})
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-4")

    custom_prompt = PromptTemplate.from_template("""
    You are acting as Dylin's executive assistant. Answer all questions as if you are Dylin speaking in the first person.

    Use the provided context to answer. If relevant, synthesize across multiple experiences.

    Include a reference link to the original case study when available, using the `source_url` metadata field.

    Speak strategically, not just descriptively. If you can't find the answer, say "Based on my available experience, I would approach it this way..." and give your best reasoning.

    Question: {question}
    Context: {context}
    Answer:
    """)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": custom_prompt,
            "document_variable_name": "context",
        },
        memory=st.session_state.memory,
        return_source_documents=True,
    )

    return qa_chain




# 🖥️ Streamlit UI
st.set_page_config(page_title="Resume Agent", layout="wide")

qa_chain = initialize_chain()

# 🔷 App Introduction Section
st.markdown(
    """
    <div style="max-width: 800px; margin: 2rem auto 1rem auto;margin-left: -1rem;">
        <h2 style="margin-bottom: 0.5rem;">Meet Dylin Webster</h2>
        <p style="font-size: 17px; line-height: 1.6;">
            Welcome to my AI-powered resume agent. Ask me anything about my leadership, strategic case studies, or experience driving outcomes in SaaS.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown('<div style="max-width: 800px; margin: 0 auto; margin-left: -1rem;">', unsafe_allow_html=True)

query = st.text_input("Ask a question:")
st.markdown('</div>', unsafe_allow_html=True)
if query:
    with st.spinner("Thinking..."):
        result = qa_chain({"question": query})

        if isinstance(result, dict):
            answer = result.get("answer", "").split("SOURCES:")[0]
            formatted_answer = answer.strip().replace("\n", "<br>")

            st.markdown(
                f"""
                <div style="background-color:#f0f4fa;
                padding:1rem 1.5rem;
                border-radius:8px;
                border-left:5px solid #003882;
                margin: 2rem auto 1rem auto;
                max-width: 1000px;
                width:100%;">
                    {formatted_answer}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ✅ Show sources if available
            sources = result.get("source_documents", [])
            if sources:
                seen_urls = set()
                st.markdown("### 🔗 Related Case Study Links")
                for doc in sources:
                    url = doc.metadata.get("source_url")
                    title = doc.metadata.get("title", "View Case Study")
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        st.markdown(f"- [{title}]({url})")
        else:
            st.markdown("⚠️ Unexpected response format.")
            st.write(result)

    
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;600;700&display=swap');
    
    body {
        background-color: #f9fbff;
    }
    
    .stMarkdown p {
        font-size: 17px;
        line-height: 1.6;
    }

    a {
        color: #003882;
        font-weight: 600;
        text-decoration: underline;
    }
    
    .stTextInput {
        max-width: 600px;
    }

    .stTextInput > div > div > input {
        height: 3rem !important;
        font-size: 16px !important;
        padding: 0.5rem 1rem !important;
        border-radius: 6px !important;
    }

    html, body, [class*="css"]  {
        font-family: 'Libre Franklin', sans-serif !important;
    }

    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #003882 !important;  /* enforce custom header color */
        font-weight: 700 !important;
    }

    .stTextInput > div > div > input {
        font-size: 20px !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .stButton button {
        background-color: #003882 !important;
        color: white !important;
        border-radius: 5px;
        font-weight: 600;
    }

    .stSpinner > div > div {
        color: #003882 !important;
    }
    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.image("docs/Headshot_Color.JPG", use_container_width=True)
    st.markdown("### Dylin Webster")
    st.markdown("📍 SaaS Exec | Customer Experience Leader")
    st.markdown("---")  # adds a horizontal line to separate sections
    
    st.markdown("#### Resume & Deck")
    st.download_button(
        label="Download Resume (PDF)",
        data=open("docs/Dylin_Webster_Resume.pdf", "rb").read(),
        file_name="Dylin_Webster_Resume.pdf",
        mime="application/pdf"
    )
    st.markdown("### View Slide Deck")
    st.markdown("[Click to open the slide deck](https://docs.google.com/presentation/d/1CEmIDMuphn13gEv67Oul1iNhFNffOnxI366OYZE5O-M/edit?usp=sharing)")

    st.markdown("#### Connect")
    st.markdown("[LinkedIn](https://linkedin.com/in/dylin-webster)")

    
from pathlib import Path
from PIL import Image

# Set the image folder path using Path
image_folder = Path("docs/Carousel")
image_files = sorted([f for f in image_folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

with st.sidebar:
    st.markdown("### 📸 Highlights")
    for image_file in image_files:
        image = Image.open(image_file)
        st.image(image, use_container_width=True)

# 🗂️ Optional: show conversation history
    
from langchain.memory import ConversationBufferMemory
    
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )



            
