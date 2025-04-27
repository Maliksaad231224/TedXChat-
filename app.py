from flask import Flask, render_template, jsonify, request
from src.helper import downlaod
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import cohere
import json
from groq import Groq

app = Flask(__name__)
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

embeddings = downlaod()
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

docsearch = PineconeVectorStore.from_existing_index(
    index_name="tedtalks",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 3})

system_prompt = (
    "You are an AI assistant that remembers past interactions with users and specializes in retrieving and summarizing TED Talks manuscripts. "
    "Use the retrieved TED Talks excerpts along with the conversation history to provide insightful and well-researched responses. "
    "Keep the conversation as detailed as possible. "
    "If the user asks for more information, generate responses based on additional TED Talk excerpts. "
    "If you don’t know the answer, say that you don’t know. Keep responses concise.\n\n"
    "Chat History:\n{chat_history}\n\nContext:\n{context}\n\n"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

co = cohere.Client(COHERE_API_KEY)

def format_chat_history(messages):
    return "\n".join([f"User: {msg.content}" if msg.type == "human" else f"AI: {msg.content}" for msg in messages])

def cohere_generate(context, question, chat_history):
    prompt_text = f"""Use the chat history and context below to answer the question.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
"""
    response = co.generate(
        model='command-r-plus',
        prompt=prompt_text,
        max_tokens=600,
        temperature=0.7
    )
    return response.generations[0].text.strip()

def deepseek_generate(context, question, chat_history):
    client = Groq(api_key=GROQ_API_KEY)
    prompt_text = f"""Use the chat history and context below to answer the question.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
"""
    try:
        completion = client.chat.completions.create(
            model="deepseek-r1-distill-qwen-32b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on TED Talks content. Return your response as a JSON object with an 'answer' key. Do not add any extra formatting or markdown."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.6,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            response_format={"type": "json_object"}
        )
        message_content = completion.choices[0].message.content
        if message_content:
            parsed = json.loads(message_content)
            return parsed.get("answer", "No 'answer' key in response.")
        else:
            return "Deepseek error: Empty response."
    except Exception as e:
        return f"Deepseek error: {str(e)}"

def llama_generate(context, question, chat_history):
    client = Groq(api_key=GROQ_API_KEY)
    prompt_text = f"""Use the chat history and context below to answer the question.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
"""
    try:
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on TED Talks content. Return your response as a JSON object with an 'answer' key. Do not add any extra formatting or markdown."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.6,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            response_format={"type": "json_object"}
        )
        message_content = completion.choices[0].message.content
        if message_content:
            parsed = json.loads(message_content)
            return parsed.get("answer", "No 'answer' key in response.")
        else:
            return "LLaMA error: Empty response."
    except Exception as e:
        return f"LLaMA error: {str(e)}"

def mistral_generate(context, question, chat_history):
    client = Groq(api_key=GROQ_API_KEY)
    prompt_text = f"""Use the chat history and context below to answer the question.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
"""
    try:
        completion = client.chat.completions.create(
            model="mistral-saba-24b",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on TED Talks content. Return your response as a JSON object with an 'answer' key. Do not add any extra formatting or markdown."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.6,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            response_format={"type": "json_object"}
        )
        message_content = completion.choices[0].message.content
        if message_content:
            parsed = json.loads(message_content)
            return parsed.get("answer", "No 'answer' key in response.")
        else:
            return "Mistral error: Empty response."
    except Exception as e:
        return f"Mistral error: {str(e)}"

def gemma_generate(context, question, chat_history):
    client = Groq(api_key=GROQ_API_KEY)
    prompt_text = f"""Use the chat history and context below to answer the question.

Chat History:
{chat_history}

Context:
{context}

Question:
{question}
"""
    try:
        completion = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on TED Talks content. Return your response as a JSON object with an 'answer' key. Do not add any extra formatting or markdown."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.6,
            max_tokens=1024,
            top_p=0.95,
            stream=False,
            response_format={"type": "json_object"}
        )
        message_content = completion.choices[0].message.content
        if message_content:
            parsed = json.loads(message_content)
            return parsed.get("answer", "No 'answer' key in response.")
        else:
            return "Gemma error: Empty response."
    except Exception as e:
        return f"Gemma error: {str(e)}"

def summarize_memory(chat_history):
    prompt_text = f"""Summarize the following conversation briefly, retaining all important insights and context:

{chat_history}
"""
    response = co.generate(
        model="command-r-plus",
        prompt=prompt_text,
        max_tokens=300,
        temperature=0.5
    )
    return response.generations[0].text.strip()


class MultiModelQAChain:
    def __init__(self, prompt_template, memory):
        self.prompt_template = prompt_template
        self.memory = memory
        self.summary = None

    def run(self, context, question, selected_model="cohere"):
        chat_messages = self.memory.load_memory_variables({})['chat_history']
        
        if len(chat_messages) >= 4:  
            formatted_chat_history = format_chat_history(chat_messages)
            summary = summarize_memory(formatted_chat_history)
            self.memory.clear()
            self.memory.save_context({"input": "[Conversation Summary]"}, {"output": summary})
        
            formatted_chat_history = ""
        else:
            formatted_chat_history = format_chat_history(chat_messages)

        if selected_model == "cohere":
            answer = cohere_generate(context, question, formatted_chat_history)
        elif selected_model == "deepseek":
            answer = deepseek_generate(context, question, formatted_chat_history)
        elif selected_model == "llama":
            answer = llama_generate(context, question, formatted_chat_history)
        elif selected_model == "mistral":
            answer = mistral_generate(context, question, formatted_chat_history)
        elif selected_model == "gemma2-9b-it":
            answer = gemma_generate(context, question, formatted_chat_history)
        else:
            answer = "Invalid model selection."

        if len(chat_messages) < 4:
            self.memory.save_context({"input": question}, {"output": answer})
        return answer


def create_retrieval_chain_with_rag(retriever, question_answering_chain):
    def chain(question, selected_model="cohere"):
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        answer = question_answering_chain.run(context, question, selected_model)
        return answer
    return chain


question_answering_chain = MultiModelQAChain(prompt, memory)
rag_chain = create_retrieval_chain_with_rag(retriever, question_answering_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/get', methods=["GET", "POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "Message is required"}), 400
    selected_model = request.form.get("selected_model", "cohere").lower()
    print(f"Selected Model: {selected_model}")  
    answer = rag_chain(msg, selected_model)

    
    return jsonify({
        "answer": answer
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
