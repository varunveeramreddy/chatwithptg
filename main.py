import fitz  # PyMuPDF for PDF text extraction
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter



def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file using PyMuPDF.
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        exit(1)


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """
    Splits the text into smaller chunks for easier query handling.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


def create_vectorstore(chunks):
    """
    Stores text chunks in a FAISS vector database using simple vector representation.
    This is just for the sake of keeping the vectorstore intact.
    """
    print("Storing chunks in FAISS...")

    # Use simple numerical encoding for chunks for FAISS
    embeddings = np.array([[len(chunk)] for chunk in chunks], dtype="float32")  # Use chunk length as "embedding"

    # Initialize FAISS index
    dimension = 1  # Simple dimension since we are using chunk lengths
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print("Chunks stored in FAISS vector database!")
    return index, chunks


def query_vectorstore(query, vectorstore, chunks):
    """
    Handles user queries by performing a basic string search on chunks.
    """
    # Perform basic keyword search: Check if query is in any chunk
    matching_chunks = [chunk for chunk in chunks if query.lower() in chunk.lower()]

    # Generate a simple response based on found matches
    if matching_chunks:
        context = "\n\n".join(matching_chunks)
        response = f"Found the following relevant text:\n{context}"
    else:
        response = "No relevant information found based on your query."

    return response


def main():
    # File path to the PDF
    pdf_path = "C:/Users/varun/OneDrive/Documents/data.pdf"

    # Step 1: Extract text
    print("Extracting text from PDF...")
    raw_text = extract_text_from_pdf(pdf_path)



    # Check if the text is empty
    if not raw_text.strip():
        print("No text extracted from the PDF. Please check the content of your PDF file.")
        return

    # Step 2: Chunk text
    print("Chunking text...")
    chunks = chunk_text(raw_text)

    # Step 3: Create vectorstore
    vectorstore, valid_chunks = create_vectorstore(chunks)

    # Step 4: Query handling
    print("You can now ask questions based on the PDF content.")
    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting...")
            break

        response = query_vectorstore(query, vectorstore, valid_chunks)
        print("\nResponse:")
        print(response)
        print("-" * 50)


if __name__ == "__main__":
    main()
