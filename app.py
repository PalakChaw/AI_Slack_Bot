import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 1. Load credentials
load_dotenv()
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")

# 2. Initialize the Slack App
app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)

# 3. Initialize the RAG "Brain"
print("Loading vector database and embedding model...")
persist_directory = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# 4. Initialize Ollama (Local LLM)
# Make sure you have Ollama running and the model pulled: ollama pull llama3
llm = OllamaLLM(model="llama3")

# 5. Setup the Retrieval Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

@app.event("app_mention")
def handle_mention(event, say):
    user_query = event["text"].split(">")[-1].strip()
    user_id = event["user"]
    
    print(f"Received query from <@{user_id}>: {user_query}")
    
    # Show "thinking" message or just proceed to search
    if not user_query:
        say(f"Hello <@{user_id}>! How can I help you with Amex engineering questions today?")
        return

    # Run the RAG chain
    try:
        response = qa_chain.invoke({"query": user_query})
        answer = response["result"]
        sources = set([doc.metadata.get("source", "Unknown") for doc in response["source_documents"]])
        
        source_text = "\n\n*Sources:* " + ", ".join(sources) if sources else ""
        say(f"Hi <@{user_id}>! {answer}{source_text}")
    except Exception as e:
        print(f"Error: {e}")
        say(f"Sorry <@{user_id}>, I encountered an error while searching the knowledge base.")

@app.message("hello")
def message_hello(message, say):
    say(f"Hey there <@{message['user']}>! Ask me anything by @mentioning me.")

if __name__ == "__main__":
    print("⚡️ Amex Knowledge Bot is running with Ollama!")
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    handler.start()
