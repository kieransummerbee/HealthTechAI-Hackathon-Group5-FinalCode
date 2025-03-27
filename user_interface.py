import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

import openai
import faiss
import numpy as np
import requests
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from flask import Flask, request, jsonify, render_template

# Create a Flask instance
app = Flask(__name__)

# Initialize OpenAI API key
import os
import requests
import spacy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

os.environ["OPENAI_API_KEY"] = "INSERT KEY"

openai.api_key  = "INSERT KEY"



def extract_entities_bert(text):
    """Extract named entities using a BERT-based model."""
    model_name = "dslim/bert-base-NER"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

    ner_results = nlp_ner(text)
    # entities = [entity["word"] for entity in ner_results if entity["entity_group"] in {"ORG", "LOC", "PER"}]
    entities = [entity["word"] for entity in ner_results ]

    cleaned_entities = []
    temp_entity = ""

    for word in entities:
        if word.startswith("##"):  # Handles BERT's word-piece tokens
            temp_entity += word[2:]  # Append without "##"
        else:
            if temp_entity:
                cleaned_entities.append(temp_entity)
            temp_entity = word

    if temp_entity:
        cleaned_entities.append(temp_entity)

    # return list(set(entities))  # Remove duplicates
    return list(set(cleaned_entities))  # Remove duplicates

import json

OPENAI_API_KEY = "your_openai_api_key"


def extract_biology_entities_openai(text):
    """Extract named entities relevant to biology using OpenAI's GPT-4."""
    prompt = f"""
    Identify and extract named entities from the following biological research question.
    Categories:
    - GENE (Genes like TP53, BRCA1)
    - PROTEIN (Proteins like Hemoglobin, Cas9)
    - DISEASE (Diseases like Cancer, Alzheimer's)
    - CHEMICAL (Compounds like ATP, Glucose)
    - BIO_PROCESS (Processes like Apoptosis, Cell Cycle)
    - ORG (Organizations, Labs, Institutions)
    - BACTERIAL_COMPONENT (Bacterial Cell Structures like Lipopolysaccharides, Peptidoglycan)

    **Text:** "{text}"

    Return only a **JSON dictionary** like this:
    {{
        "GENE": ["Gene1", "Gene2"],
        "PROTEIN": ["Protein1", "Protein2"],
        "DISEASE": ["Disease1", "Disease2"],
        "CHEMICAL": ["Chemical1", "Chemical2"],
        "BIO_PROCESS": ["Process1", "Process2"],
        "ORG": ["Organization1", "Organization2"],
        "BACTERIAL_COMPONENT": ["Component1", "Component2"]
    }}

    If no entities are found, return an empty dictionary.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Keeps output consistent
        )

        raw_output = response["choices"][0]["message"]["content"].strip()

        # Debug: Print raw output
        print("RAW OUTPUT FROM GPT:\n", raw_output)

        raw_output = str(raw_output).replace('```json', '').replace('```','').replace('\n','')

        # Parse JSON output
        entities = json.loads(raw_output)

        return entities
    except json.JSONDecodeError:
        print(" Error: Could not parse JSON. Raw output:", raw_output)
        return {"GENE": [], "PROTEIN": [], "DISEASE": [], "CHEMICAL": [], "BIO_PROCESS": [], "ORG": [],
                "BACTERIAL_COMPONENT": []}


# Function to query Semantic Scholar API to get paper metadata and abstract
def fetch_paper_abstract(query):
    search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': query,  # Your query string
        'limit': 15,
        "fields": "title,abstract"
    }

    api_key = "CGcH1D5EkY1KvkzSny3jZ9sjUncb3bIk2zqGCzwl"  # Replace with the actual API key

    # Define headers with API key
    headers = {"x-api-key": api_key}

    # response = requests.get(search_url, params=params, headers=headers)
    response = requests.get(search_url, params=params)
    data = response.json()

    papers = []
    if 'data' in data:
        for paper in data['data']:
            title = paper.get('title', 'No title')
            abstract = paper.get('abstract', 'No abstract available')
            papers.append({'title': title, 'abstract': abstract})
    return papers

def clean_text(text):
    # Remove irrelevant sections, ads, etc. (simplified for this example)
    clean_text = str(text).replace('\n', ' ').strip()
    return clean_text



# Load spaCy NER model
nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en_core_web_lg")

def extract_entities(text):
    """Extract named entities from a text query."""
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return list(set(entities))

# 5. Querying with a specific prompt
def query_article(query, faiss_index, chunks, embedding_model):
    query_embedding = embedding_model.embed_documents([query])[0]  # Get embedding for the query

    # Perform semantic search in FAISS
    query_embedding = np.array(query_embedding).astype('float32')
    _, indices = faiss_index.search(query_embedding.reshape(1, -1), k=3)  # Get top 3 relevant chunks

    # Return the top k relevant chunks
    relevant_chunks = [chunks[i] for i in indices[0]]

    return relevant_chunks

from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI


def answer_question(summary, question):
    chat = ChatOpenAI(model="gpt-4")

    conversation = [
        SystemMessage(content="You are a research expert providing precise answers based on research."),
        HumanMessage(
            content=f"Based on this research summary, answer the following question in one sentence:\n\nSummary: {summary}\n\nQuestion: {question}")
    ]

    response = chat(conversation)
    return response.content.strip()

def get_entity_openai(query):

    entities = extract_biology_entities_openai(query)
    query_list = ''
    for value in entities.values():
        if len(value) > 0:
            query_list = query_list + str(value[0]) + ' '

    return query_list

import re

def match_regex(sentence):
    pattern = r"\b(?=\S*[a-zA-Z])(?=\S*\d)(?=\S*\.)[a-zA-Z0-9.]+\b"


    all_matched_words = ""
    matches = re.findall(pattern, sentence)
    for each_m in matches:
        all_matched_words = all_matched_words + str(each_m) + " "

    return all_matched_words.strip()

def get_entity_spacy(query):
    entities = extract_entities(query)
    # entities = extract_entities_bert(query)
    query_list = " ".join(entities)

    return query_list

@app.route('/')
def read_form():
    return render_template("index.html")

@app.route("/submit/", methods=['POST'])
def summarize_with_gpt():

    query = request.form["questionText"]

    # query_list = get_entity_spacy(query)
    query_list = get_entity_openai(query)

    if query_list.strip() == "":
        query_list = match_regex(query_list)


    papers = fetch_paper_abstract(query_list.strip())
    # papers = fetch_paper_abstract(query)


    if papers:
        print(f"Found {len(papers)} papers related to the query:")
        for i, paper in enumerate(papers):
            print(f"{i+1}. Title: {paper['title']}")
            print(f"   Abstract: {paper['abstract']}")  # Show a preview of the abstract
    else:
        print("No relevant papers found.")
        return "No relevant papers found."


    abstract_chunks = []
    for paper in papers:
        cleaned_abstract = clean_text(paper['abstract'])
        abstract_chunks.append(cleaned_abstract)

    # 2. Chunking the abstract into smaller parts (if needed)
    # For this example, we'll just use the entire abstract as a chunk
    # chunks = [cleaned_abstract]

    # 3. Embedding the chunks using OpenAI embeddings
    embedding_model = OpenAIEmbeddings()

    # Convert each chunk into a vector (embedding)
    embeddings = embedding_model.embed_documents(abstract_chunks)  # Use embed_documents method

    # 4. Creating a FAISS index for semantic search
    embedding_dimension = len(embeddings[0])  # Dimension of the embedding vectors
    faiss_index = faiss.IndexFlatL2(embedding_dimension)  # L2 distance metric for similarity search

    # Convert embeddings into numpy array
    embedding_matrix = np.array(embeddings).astype('float32')

    # Add embeddings to the FAISS index
    faiss_index.add(embedding_matrix)



    # 6. Example: Querying the article (this is just the abstract, but you can use more)
    relevant_chunks = query_article(query, faiss_index, abstract_chunks, embedding_model)

    # Output relevant chunks from the article
    print("\nRelevant sections of the article:")
    for i, chunk in enumerate(relevant_chunks):
        print(f"{i + 1}. {chunk}")


    # Combine the relevant chunks into a larger context for summarization
    context = " ".join(relevant_chunks)

    # dddasd = f"Reply with a sentence for this question that {query} :\n\n{context}"
    # print('-----')
    # print(dddasd)
    # print('-----')

    # Summarize using GPT-4 via the chat API (corrected method)
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Specify GPT-4 or any other available model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Reply with a sentence for this question {query} :\n\n{context}"}
        ],
        max_tokens=100
    )

    result = str(response['choices'][0]['message']['content'].strip())

    langchain_answer = answer_question(context, query)

    return render_template("index.html", text_result='Result 1: ' +result, papers_list= papers, number_of_paper =len(papers),
                           question="Question : "+str(query), langchain_answer ='Result 2: ' + langchain_answer)
                           # question="Question : "+str(query) +" - " + str(query_list), langchain_answer ='Result 2: ' + langchain_answer)


if __name__ == "__main__":
    app.run(debug=True)
