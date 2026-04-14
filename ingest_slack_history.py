import os
import time
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# 1. Load credentials
load_dotenv()
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
client = WebClient(token=SLACK_BOT_TOKEN)

# 2. Initialize Embeddings and Vector DB
persist_directory = "chroma_db"
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def fetch_channels():
    """List all public channels the bot has access to."""
    try:
        result = client.conversations_list(types="public_channel")
        return result["channels"]
    except SlackApiError as e:
        print(f"Error fetching channels: {e}")
        return []

def fetch_messages(channel_id, limit=100):
    """Fetch recent messages from a specific channel."""
    try:
        print(f"Fetching messages for channel: {channel_id}...")
        result = client.conversations_history(channel=channel_id, limit=limit)
        messages = result["messages"]
        
        # Filter out bot messages and subtype messages (like 'channel_join')
        clean_messages = []
        for msg in messages:
            if "bot_id" not in msg and "text" in msg and len(msg["text"]) > 10:
                clean_messages.append({
                    "text": msg["text"],
                    "ts": msg["ts"],
                    "user": msg.get("user", "Unknown")
                })
        return clean_messages
    except SlackApiError as e:
        print(f"Error fetching messages: {e}")
        return []

def ingest_to_db(messages, channel_name):
    """Convert messages to documents and add to ChromaDB."""
    if not messages:
        return
    
    docs = []
    for m in messages:
        # Create a deep link to the Slack message
        # Note: You'd need the Workspace ID for a perfect link, but we'll use channel name for now.
        source_info = f"Slack Channel: #{channel_name}"
        
        doc = Document(
            page_content=m["text"],
            metadata={"source": source_info, "ts": m["ts"], "user": m["user"]}
        )
        docs.append(doc)
    
    print(f"Adding {len(docs)} messages from #{channel_name} to the database...")
    vectordb.add_documents(docs)
    print("Ingestion successful!")

if __name__ == "__main__":
    channels = fetch_channels()
    
    if not channels:
        print("No public channels found. Make sure the bot is invited to the channels!")
    else:
        print(f"Found {len(channels)} channels.")
        for channel in channels:
            # We only ingest channels the bot is a member of
            if channel["is_member"]:
                msgs = fetch_messages(channel["id"])
                ingest_to_db(msgs, channel["name"])
            else:
                print(f"Skipping #{channel['name']} (Bot is not a member. Run /invite @AmexKnowledgeBot in Slack first).")

    print("\n✅ All available Slack history has been indexed into ChromaDB.")
