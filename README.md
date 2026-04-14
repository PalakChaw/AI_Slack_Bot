# AI_Slack_Bot
Creating an AI AMEX Slack bot with RAG (Retrieval-Augmented Generation).

## Setup
1.  **Clone the repository**:
    ```bash
    git clone https://github.com/PalakChaw/AI_Slack_Bot.git
    cd AI_Slack_Bot
    ```

2.  **Create a virtual environment and install dependencies**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up your environment variables**:
    Create a `.env` file with the following variables:
    ```bash
    SLACK_BOT_TOKEN=xoxb-your-bot-token
    SLACK_APP_TOKEN=xapp-your-app-token
    SLACK_SIGNING_SECRET=your-signing-secret
    ```

4.  **Run Ollama**:
    Ensure you have [Ollama](https://ollama.ai/) installed and the `llama3` model pulled:
    ```bash
    ollama pull llama3
    ```

5.  **Ingest data**:
    Run the ingestion scripts to populate your vector database:
    ```bash
    python ingest_knowledge.py
    python ingest_slack_history.py
    ```

6.  **Run the bot**:
    ```bash
    python app.py
    ```
