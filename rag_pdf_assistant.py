import getpass
import os
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Ask for LangChain API key if not set in environment variables
if not os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = getpass.getpass(
        "Please enter your LangChain API key: ")

# Ask for OpenAI API key if not set in environment variables
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass(
        "Please enter your OpenAI API key: ")


# 1. Extract Text from the PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text


# 2. Split Text into Chunks
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = text_splitter.split_text(text)
    return chunks


# 3. Create a Retriever Using a Vector Database
def create_retriever(chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 6})
    return retriever


# 4. Combine all relevant chunks
def combine_chunks(chunks):
    return "\n\n".join(chunk.page_content for chunk in chunks)


# 5. Summarize Retrieved Text
def summarize_pdf(question, retriever, llm="gpt-4"):
    llm = ChatOpenAI(model=llm)

    rag_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """

    prompt_template = PromptTemplate(
        template=rag_prompt, input_variables=["context", "question"])

    # Set up the retrieval and generation chain
    rag_chain = (
        {"context": retriever | combine_chunks, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    result = []
    for chunk in rag_chain.stream(question):
        result.append(chunk)
    return "".join(result)


if __name__ == "__main__":
    # Ask user for the PDF file path
    pdf_path = input("\nPlease enter the path to the PDF file: ")
    text = extract_text_from_pdf(pdf_path)

    # Split the text into chunks
    chunks = split_text_into_chunks(text, chunk_size=1000, overlap=200)

    # Create retriever
    retriever = create_retriever(chunks)

    # Ask user for the question to be answered
    question = input("\nPlease enter the question for the model: ")

    # Summarize the PDF based on the user's question
    result = summarize_pdf(question, retriever, "gpt-4")

    # Print the result
    print("\n\n", result)
