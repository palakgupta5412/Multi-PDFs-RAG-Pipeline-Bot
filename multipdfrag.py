# Step 1 : initial setup 
from groq import Groq
import os 
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np 
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Step 2 : Read PDF ->
def readPDFs(folder) :
    documents = [] 

    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            reader = PdfReader(path)

            for i,page in enumerate(reader.pages):
                text = page.extract_text()
                if text :
                    documents.append({
                        "text" : text ,
                        "page" : i+1 ,
                        "source" : file 
                    })

    return documents

# Step 3 : Chunk Text from PDF ->

def createChunks(documents , size=500 , overlap = 100):

    chunks=[]

    # This we are doing so that the context of one sentence does not get cut and go half into one chunk and half into another
    for doc in documents :
        sentences = sent_tokenize(doc["text"])
        curr_chunk = ""
        
        for sentence in sentences :
            if not sentence.strip():
                continue
            if len(curr_chunk) + len(sentence) < size :
                curr_chunk += sentence+" "
            else:
                chunks.append({
                    "text" : curr_chunk,
                    "page" : doc["page"],
                    "source" : doc["source"]
                })

                curr_chunk = sentence

            if curr_chunk :
                chunks.append({
                    "text": curr_chunk,
                    "page": doc["page"],
                    "source": doc["source"]
                })

    return chunks

# Step 4 : re-defining user query to better improve search ->
def rewriteQuery(query) :
    prompt = f"""
    Rewrite the question to make it slightly clearer.
    Keep it short (max 10 words).    Original question:
    {query}

    Improved search query:
    """

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()

# Step 5 : Loading docs 
docs = readPDFs('docs')
chunks = createChunks(docs)
print("Length of chunks : " , len(chunks))

# Step 6 : Create Embeddings ->
embeddingModel = SentenceTransformer("all-MiniLM-L6-v2")
texts = [ch["text"] for ch in chunks]
print("Retrieved Chunks texts is: " , texts)
embeddings = embeddingModel.encode(texts)

# Step 7 : Create Index ->
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

while True : 
    query = input("Enter your query: ")
    if(query=="exit"):
        break
    improved_query = rewriteQuery(query)
    query_embedding = embeddingModel.encode([improved_query])
    distances , indices = index.search(query_embedding , k=5)

    # Shows how close the query is to each chunk
    for i, idx in enumerate(indices[0]):
        print("Score:", distances[0][i])

    relevant_chunks = [ chunks[i] for i in indices[0]]

    # context = "\n".join(chnk["text"] for chnk in relevant_chunks)
    context = "" 
    for i,chunk in enumerate(relevant_chunks):
        # To prevent LLM overflow 
        if len(chunk["text"])+len(context)<2000 :
            context+= f"Source: {chunk['source']} Page: {chunk['page']} \n"
            context+= f"{chunk['text']} \n"
    prompt = f"""

    You are answering questions using the provided context from documents.

    Use the context to answer the question clearly.
    If the answer is partially available, explain using the available information.
    Context : 
    {context}

    Question : 
    {query}
    """

    print("Answer : ")
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    fullRes = "" 

    for chunk in stream :
        content = chunk.choices[0].delta.content

        if content :
            print(content , end="" , flush=True)
            fullRes += content

    print("")

    print('\nSources : ')
    for chnk in relevant_chunks:
        print(f"{chnk['source']} : Page {chnk['page']}")





