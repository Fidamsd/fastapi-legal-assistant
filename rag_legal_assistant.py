# from fastapi import FastAPI, Form, UploadFile, File
# from fastapi.responses import HTMLResponse
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# import os
# import shutil

# os.environ["GOOGLE_API_KEY"] = "AIzaSyBRAKfYEnbImVZOEESX7KuIA8Op5mWI9js"

# app = FastAPI()

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash-lite-preview-06-17",
#     temperature=0
# )
# qa_chain = None

# @app.get("/", response_class=HTMLResponse)
# async def home():
#     html_content = """
#     <!DOCTYPE html>
#     <html>
#     <head>
#         <title>FastAPI Gemini Q&A</title>
#     </head>
#     <body>
#         <h2>Upload PDF Document</h2>
#         <form action="/upload" enctype="multipart/form-data" method="post">
#             <input type="file" name="file">
#             <button type="submit">Upload</button>
#         </form>

#         <h2>Ask a Question</h2>
#         <form action="/ask" method="post">
#             <input type="text" name="question" placeholder="Enter your legal question">
#             <button type="submit">Ask</button>
#         </form>
#     </body>
#     </html>
#     """
#     return HTMLResponse(content=html_content)

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     folder_path = "law_docs"
#     os.makedirs(folder_path, exist_ok=True)
#     file_path = os.path.join(folder_path, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     documents = []
#     loader = PyPDFLoader(file_path)
#     documents.extend(loader.load())
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     texts = text_splitter.split_documents(documents)

#     global qa_chain
#     db = FAISS.from_documents(texts, embeddings)
#     retriever = db.as_retriever()

#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever
#     )

#     return HTMLResponse(content="<p>✅ File uploaded and processed. Go back and ask questions now.</p>")

# @app.post("/ask")
# async def ask_question(question: str = Form(...)):
#     global qa_chain
#     if qa_chain is None:
#         return HTMLResponse(content="<p>⚠️ Please upload a document first.</p>")
#     answer = qa_chain.run(question)
#     return HTMLResponse(content=f"<p>🪐 Answer: {answer}</p><p><a href='/'>Go Back</a></p>")


# ------------------ rag_legal_assistant.py ------------------


import os
import shutil
from datetime import datetime
from fastapi import FastAPI, Request, Form, UploadFile, File, Depends
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime, LargeBinary, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from passlib.context import CryptContext
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import PlainTextResponse
from fastapi_utils.tasks import repeat_every
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytz



app = FastAPI()

# Allow your Netlify frontend to access this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://clinquant-toffee-61a281.netlify.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup
os.environ["GOOGLE_API_KEY"] = "AIzaSyBcIy4IuRD_Ih-umLWKucjmkqLPVsQIi3Q"
DATABASE_URL = "postgresql://postgres:1212@localhost:5432/law_assistant_db"

app = FastAPI()
#app.add_middleware(SessionMiddleware, secret_key="super-secret-key")
app.add_middleware(
    SessionMiddleware,
    secret_key="super-secret-key",
    same_site="none",       # ✅ allow cookies across domains
    https_only=True         # ✅ cookies will work on Netlify HTTPS
)

templates = Jinja2Templates(directory="templates")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    chats = relationship("Chat", back_populates="owner")
    documents = relationship("Document", back_populates="user")

class Chat(Base):
    __tablename__ = "chats"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    question = Column(Text)
    answer = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="chats")

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    content = Column(LargeBinary, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="documents")

Base.metadata.create_all(bind=engine)

# Gemini Setup
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
user_retrievers = {}

def get_user_by_email(db, email):
    return db.query(User).filter(User.email == email).first()

def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

@app.get("/register")
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
def register(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    if get_user_by_email(db, email):
        return HTMLResponse("Email already registered. <a href='/login'>Login</a>")
    user = User(username=username, email=email, hashed_password=get_password_hash(password))
    db.add(user)
    db.commit()
    return RedirectResponse("/login", status_code=303)

@app.get("/login")
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    db = SessionLocal()
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return HTMLResponse("Invalid credentials. <a href='/login'>Try again</a>")
    request.session["user_id"] = user.id

    # ✅ Check if user has any chats, if not insert welcome message
    existing_chats = db.query(Chat).filter_by(user_id=user.id).first()
    if not existing_chats:
        pakistan_tz = pytz.timezone("Asia/Karachi")
        welcome_time = datetime.now(pakistan_tz)   # ✅ Local time
        welcome_message = Chat(
            user_id=user.id,
            question="",
            answer="👋 Hello! I'm your Legal Assistant AI. How can I help you with your legal questions today?",
            created_at=welcome_time                 # ✅ Save with correct timezone
        )
        db.add(welcome_message)
        db.commit()

    return RedirectResponse("/", status_code=303)

# @app.get("/logout")
# def logout(request: Request):
#     request.session.clear()
#     return RedirectResponse("/login", status_code=303)

#Delete chat when user logout
@app.get("/logout")
def logout(request: Request):
    user_id = request.session.get("user_id")
    if user_id:
        db = SessionLocal()
        db.query(Chat).filter_by(user_id=user_id).delete()
        db.commit()
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse("/login", status_code=303)
    db = SessionLocal()
    user = db.query(User).filter_by(id=user_id).first()
    chats = db.query(Chat).filter_by(user_id=user_id).order_by(Chat.created_at).all()
    pakistan_tz = pytz.timezone("Asia/Karachi")
    for msg in chats:
        msg.created_at = msg.created_at.astimezone(pakistan_tz)
    return templates.TemplateResponse(
    "index.html",
    {
        "request": request,
        "user": user,
        "chat": {"messages": chats},
        "pytz": pytz,
        
    }
)
@app.post("/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    user_id = request.session.get("user_id")
    if not user_id:
        return RedirectResponse("/login", status_code=303)
    content = await file.read()
    db = SessionLocal()
    document = Document(filename=file.filename, content=content, user_id=user_id)
    db.add(document)
    db.commit()
    user_retrievers.pop(user_id, None)  # clear cache so new retriever will be rebuilt on next ask

    # Determine file type from extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    file_type = "PDF Document" if file_ext == ".pdf" else "Word Document" if file_ext == ".docx" else "Unknown File"

    # Save as a USER message (left side)
    message = Chat(
        user_id=user_id,
        question=f"File uploaded: {file.filename} ({file_type})",  # shown as user message
        answer=""
    )
    db.add(message)
    db.commit()

    return RedirectResponse("/", status_code=303)



# @app.post("/ask")
# async def ask_question(request: Request, question: str = Form(...)):
#     user_id = request.session.get("user_id")
#     if not user_id:
#         return RedirectResponse("/login", status_code=303)

#     db = SessionLocal()
#     retriever = user_retrievers.get(user_id)

#     if not retriever:
#         documents = db.query(Document).filter_by(user_id=user_id).all()
#         if not documents:
#             answer = "⚠️ Please upload at least one document first."
#         else:
#             all_chunks = []
#             splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#             for doc in documents:
#                 # Save file to a temp location
#                 ext = os.path.splitext(doc.filename)[1].lower()
#                 temp_path = f"temp{ext}"
#                 with open(temp_path, "wb") as f:
#                     f.write(doc.content)

#                 # Select appropriate loader
#                 try:
#                     if ext == ".pdf":
#                         loader = PyPDFLoader(temp_path)
#                     elif ext == ".docx":
#                         loader = Docx2txtLoader(temp_path)
#                     else:
#                         continue  # skip unsupported file types

#                     chunks = splitter.split_documents(loader.load())
#                     all_chunks.extend(chunks)
#                 finally:
#                     if os.path.exists(temp_path):
#                         os.remove(temp_path)

#             if all_chunks:
#                 db_local = FAISS.from_documents(all_chunks, embeddings)
#                 retriever = db_local.as_retriever()
#                 user_retrievers[user_id] = retriever
#             else:
#                 answer = "⚠️ No valid documents found to process."
#                 chat = Chat(user_id=user_id, question=question, answer=answer)
#                 db.add(chat)
#                 db.commit()
#                 return RedirectResponse("/", status_code=303)

#     if retriever:
#         qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
#         answer = qa_chain.run(question)
#     else:
#         answer = "⚠️ No retriever available."

#     chat = Chat(user_id=user_id, question=question, answer=answer)
#     db.add(chat)
#     db.commit()
#     return RedirectResponse("/", status_code=303)


# @app.post("/ask", response_class=PlainTextResponse)
# async def ask_question(request: Request, question: str = Form(...)):
#     user_id = request.session.get("user_id")
#     if not user_id:
#         return "Please login first."

#     db = SessionLocal()
#     retriever = user_retrievers.get(user_id)

#     if not retriever:
#         documents = db.query(Document).filter_by(user_id=user_id).all()
#         if not documents:
#             answer = "⚠️ Please upload at least one document first."
#         else:
#             all_chunks = []
#             splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

#             for doc in documents:
#                 ext = os.path.splitext(doc.filename)[1].lower()
#                 temp_path = f"temp{ext}"
#                 with open(temp_path, "wb") as f:
#                     f.write(doc.content)

#                 try:
#                     if ext == ".pdf":
#                         loader = PyPDFLoader(temp_path)
#                     elif ext == ".docx":
#                         loader = Docx2txtLoader(temp_path)
#                     else:
#                         continue

#                     chunks = splitter.split_documents(loader.load())
#                     all_chunks.extend(chunks)
#                 finally:
#                     if os.path.exists(temp_path):
#                         os.remove(temp_path)

#             if all_chunks:
#                 db_local = FAISS.from_documents(all_chunks, embeddings)
#                 retriever = db_local.as_retriever()
#                 user_retrievers[user_id] = retriever
#             else:
#                 answer = "⚠️ No valid documents found to process."
#                 chat = Chat(user_id=user_id, question=question, answer=answer)
#                 db.add(chat)
#                 db.commit()
#                 return answer

#     if retriever:
#         qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
#         answer = qa_chain.run(question)
#     else:
#         answer = "⚠️ No retriever available."

#     chat = Chat(user_id=user_id, question=question, answer=answer)
#     db.add(chat)
#     db.commit()
#     return answer

#////////////////
from fastapi import FastAPI, Request, Form
from fastapi.responses import PlainTextResponse
from datetime import datetime
import os

@app.post("/ask", response_class=PlainTextResponse)
async def ask_question(request: Request, question: str = Form(...)):
    user_id = request.session.get("user_id")
    if not user_id:
        return "Please login first."

    db = SessionLocal()
    retriever = user_retrievers.get(user_id)

    # Update last activity and reset away/goodbye message flags
    request.session["last_activity"] = datetime.utcnow().isoformat()
    if not hasattr(app.state, "sessions"):
        app.state.sessions = {}
    if user_id not in app.state.sessions:
        app.state.sessions[user_id] = {
            "last_activity": request.session["last_activity"],
            "sent_away_message": False,
            "sent_goodbye_message": False
        }
    else:
        app.state.sessions[user_id]["last_activity"] = request.session["last_activity"]
        app.state.sessions[user_id]["sent_away_message"] = False
        app.state.sessions[user_id]["sent_goodbye_message"] = False

    if not retriever:
        documents = db.query(Document).filter_by(user_id=user_id).all()
        if not documents:
            answer = "⚠️ Please upload at least one document first."
            chat = Chat(user_id=user_id, question=question, answer=answer)
            db.add(chat)
            db.commit()
            return answer  # ✅ important line added here

        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for doc in documents:
            ext = os.path.splitext(doc.filename)[1].lower()
            temp_path = f"temp{ext}"
            with open(temp_path, "wb") as f:
                f.write(doc.content)

            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(temp_path)
                elif ext == ".docx":
                    loader = Docx2txtLoader(temp_path)
                else:
                    continue

                chunks = splitter.split_documents(loader.load())
                all_chunks.extend(chunks)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        if all_chunks:
            db_local = FAISS.from_documents(all_chunks, embeddings)
            retriever = db_local.as_retriever()
            user_retrievers[user_id] = retriever
        else:
            answer = "⚠️ No valid documents found to process."
            chat = Chat(user_id=user_id, question=question, answer=answer)
            db.add(chat)
            db.commit()
            return answer

    if retriever:
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        answer = qa_chain.run(question)
    else:
        answer = "⚠️ No retriever available."

    chat = Chat(user_id=user_id, question=question, answer=answer)
    db.add(chat)
    db.commit()
    return answer




#///////////////
@app.on_event("startup")
@repeat_every(seconds=30)
def check_inactive_users():
    if not hasattr(app.state, "sessions"):
        app.state.sessions = {}

    db = SessionLocal()
    now = datetime.utcnow()

    for user_id, session_data in list(app.state.sessions.items()):
        last_activity = datetime.fromisoformat(session_data["last_activity"])
        sent_away_message = session_data.get("sent_away_message", False)
        sent_goodbye_message = session_data.get("sent_goodbye_message", False)

        inactivity_duration = (now - last_activity).total_seconds()

        # 2 min inactivity: send away message
        if not sent_away_message and inactivity_duration > 30:
            auto_message = Chat(
                user_id=user_id,
                question="",
                answer="Are you still there? Let me know if you need further legal assistance."
            )
            db.add(auto_message)
            db.commit()
            session_data["sent_away_message"] = True
            print(f"Sent away message to user_id {user_id}")

        # 4 min inactivity: send goodbye and clear session
        if not sent_goodbye_message and inactivity_duration > 60:
            goodbye_message = Chat(
                user_id=user_id,
                question="",
                answer="Session ended due to inactivity. Thank you for using Law Assistant. Have a wonderful day! 🌿"
            )
            db.add(goodbye_message)
            db.commit()
            session_data["sent_goodbye_message"] = True
            print(f"Sent goodbye message to user_id {user_id}")

            # Remove session safely
            del app.state.sessions[user_id]

