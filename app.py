import os
import getpass
import PyPDF2
from flask import Flask, request, redirect, render_template_string, flash, url_for
from werkzeug.utils import secure_filename

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Ensure API keys are set (or you can set them as environment variables before running)
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("Please enter your LangChain API key: ")
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Please enter your OpenAI API key: ")

# Define your helper functions from your original code
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def create_retriever(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever

def combine_chunks(chunks):
    # Assumes each chunk has an attribute 'page_content'
    return "\n\n".join(chunk.page_content for chunk in chunks)

def summarize_pdf(question, retriever, llm="gpt-4"):
    llm_model = ChatOpenAI(model=llm)
    rag_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt_template = PromptTemplate(template=rag_prompt, input_variables=["context", "question"])
    
    # Set up the chain using a functional pipeline
    rag_chain = (
        {"context": retriever | combine_chunks, "question": RunnablePassthrough()}
        | prompt_template
        | llm_model
        | StrOutputParser()
    )
    
    result = []
    for chunk in rag_chain.stream(question):
        result.append(chunk)
    return "".join(result)

# Set up the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SECRET_KEY'] = 'secret!'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# HTML templates are defined inline for simplicity.
upload_page = '''
<!doctype html>
<html>
<head>
  <title>Upload PDF for Q&amp;A</title>
</head>
<body>
  <h1>Upload PDF and Ask a Question</h1>
  {% with messages = get_flashed_messages() %}
    {% if messages %}
      <ul>
        {% for message in messages %}
          <li>{{ message }}</li>
        {% endfor %}
      </ul>
    {% endif %}
  {% endwith %}
  <form method="post" enctype="multipart/form-data">
    <label for="pdf_file">PDF File:</label>
    <input type="file" name="pdf_file" id="pdf_file"><br><br>
    <label for="question">Question:</label>
    <input type="text" name="question" id="question"><br><br>
    <input type="submit" value="Submit">
  </form>
</body>
</html>
'''

result_page = '''
<!doctype html>
<html>
<head>
  <title>PDF Q&amp;A Result</title>
</head>
<body>
  <h1>Answer</h1>
  <p>{{ answer }}</p>
  <a href="{{ url_for('index') }}">Back</a>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ensure a file is provided
        if 'pdf_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['pdf_file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Get the question from the form
            question = request.form.get('question')
            if not question:
                flash("Please provide a question.")
                return redirect(request.url)
            
            # Process the PDF using your functions
            text = extract_text_from_pdf(filepath)
            chunks = split_text_into_chunks(text)
            retriever = create_retriever(chunks)
            answer = summarize_pdf(question, retriever, llm="gpt-4")
            
            # Clean up the uploaded file if desired
            os.remove(filepath)
            
            return render_template_string(result_page, answer=answer)
    return render_template_string(upload_page)

if __name__ == '__main__':
    app.run(debug=True)