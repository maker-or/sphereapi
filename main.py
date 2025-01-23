import requests
import os
import shutil
import pinecone
import google.generativeai as genai

from flask import Flask,request
from flask_cors import CORS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

app = Flask(__name__)
CORS(app)

@app.route('/check',methods=['GET'])
def check():
    return "chelc"

@app.route('/', methods=['POST'])
def index():
    print('request acepted')
    print('qrgs:',request.form)
    fileUrl = request.form.get('url')
    print('fileUrl:',fileUrl)
    uploadthing_url = (
        fileUrl
    )
    folder_t1 = "t1"
    folder_0 = "0"
    local_file_path = os.path.join(folder_t1, "downloaded_file.pdf")
    os.makedirs(folder_t1, exist_ok=True)
    os.makedirs(folder_0, exist_ok=True)

    def download_file_from_url(url, save_path):
        response = requests.get(url)
        if response.status_code == 200:
            with open(save_path, "wb") as file:
                file.write(response.content)
            print(f"File downloaded to {save_path}")
        else:
            raise Exception(
                f"Failed to download file. Status code: {response.status_code}"
            )

    download_file_from_url(uploadthing_url, local_file_path)

    # Convert PDF to Markdown
    converter = PdfConverter(artifact_dict=create_model_dict())
    rendered = converter(local_file_path)
    MD, _, images = text_from_rendered(rendered)
    print("Converted text is:", MD)

    # Load the markdown content directly
    markdown_document = MD

    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 400
    BATCH_SIZE = 100

    GOOGLE_API_KEY = 'AIzaSyC-yW2eWHjJFXvmcyv01sSOyCU9g8JNy1g'
    PINECONE_API_KEY = '90d8a0fc-3d7f-4782-b675-d3b84d1f6956'
    PINECONE_ENVIRONMENT = 'us-east-1'
    INDEX_NAME = "ol"

    # Create a list of books from the markdown content
    books = [{"title": "Converted Document", "content": markdown_document}]
    data_for_upsert = []

    genai.configure(api_key=GOOGLE_API_KEY)
    for book in books:
        book_title = book["title"]
        markdown_document = book["content"]

        # Split the document
        headers_to_split_on = [("#", "Header 1")]
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        md_header_splits = markdown_splitter.split_text(markdown_document)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(md_header_splits)

        # === Prepare Data for Pinecone Section ===
        for i, split in enumerate(splits):
            text = split.page_content  # Extract the text content

            try:
                # Generate embeddings
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=[text]
                )
                embedding = response['embedding'][0]  # Adjust if needed

                # Create metadata dictionary
                metadata = {
                    'text': text,
                    'source': f'document_chunk_{i}',
                    'book': book_title,  # Add the book title to metadata
                    'page_number': split.metadata.get('page', '')
                }

                # Append to upsert data
                data_for_upsert.append({
                    'id': f"{book_title}_chunk_{i}",
                    'values': embedding,
                    'metadata': metadata
                })

            except Exception as e:
                print(f"Error generating embedding for chunk {i} from "
                      f"{book_title}: {e}")

    # === Pinecone Upsert Section ===
    try:
        # Initialize Pinecone
        pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

        # Connect to the index
        index = pc.Index(INDEX_NAME)

        # Batch upsert
        for i in range(0, len(data_for_upsert), BATCH_SIZE):
            batch = data_for_upsert[i:i + BATCH_SIZE]
            index.upsert(vectors=batch)
            print(f"Upserted batch {i // BATCH_SIZE + 1} of "
                  f"{len(data_for_upsert) // BATCH_SIZE + 1}")

    except Exception as e:
        print(f"Error during Pinecone upsert: {e}")

    # Clean up
    if os.path.exists(local_file_path):
        shutil.rmtree(folder_t1)
        shutil.rmtree(folder_0)
        print(f"File {local_file_path} has been deleted.")
    else:
        print(f"File {local_file_path} does not exist.")

    return "Hello World"


if __name__ == "__main__":
    app.run(debug=False, port=8800)
