import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# 1. Define expanded sample knowledge (Mocking Slack/Confluence data)
sample_data = [
    # EPCC & Migrations
    {
        "content": "To resolve EPCC migration error 403, ensure that the service principal has the 'Reader' role assigned in the dev environment. Check the IAM policy in the Azure portal.",
        "source": "Slack #epcc-migration-help"
    },
    {
        "content": "The EPCC migration to the new Kubernetes cluster requires updating the 'ingress-controller' to version 2.4.1. Older versions will cause intermittent 502 Bad Gateway errors.",
        "source": "Confluence: Infrastructure Guide"
    },
    # API & Integration
    {
        "content": "The standard API timeout for internal Amex services is 30 seconds. For larger payloads, use the async-endpoint pattern described in Confluence page ID 8821.",
        "source": "Confluence: Engineering Standards"
    },
    {
        "content": "If you see 'Unauthorized' in the Card-Services-API, check if your OAuth token has the 'card-read' scope. Tokens expire every 60 minutes and require a refresh token flow.",
        "source": "Slack #card-services-api"
    },
    {
        "content": "The Customer-Profile-Service uses GraphQL. If you get a 'Field not found' error, ensure you are using the v3 schema. v2 was deprecated last quarter.",
        "source": "Slack #api-announcements"
    },
    # Databases & Caching
    {
        "content": "When connecting to the Postgres production database, always use the read-replica for SELECT queries to avoid putting load on the primary node. Port is 5433.",
        "source": "Confluence: DB Best Practices"
    },
    {
        "content": "Redis cache eviction policy is set to 'volatile-lru'. If you are losing data unexpectedly, check if your keys have an expiration (TTL) set correctly.",
        "source": "Slack #platform-infra"
    },
    # Authentication & Security
    {
        "content": "All internal apps must use the 'Amex-SSO' plugin. If your app is stuck in a redirect loop, clear your browser cookies or check the 'redirect_uri' in the app registration.",
        "source": "Slack #security-ops"
    },
    {
        "content": "Secrets should never be stored in plaintext. Use the 'Amex-Vault' service to fetch secrets at runtime. Access requires a valid 'App-ID' and 'Secret-ID'.",
        "source": "Confluence: Security Compliance"
    },
    # CI/CD & DevOps
    {
        "content": "Jenkins builds failing with 'No space left on device' are usually due to old Docker images. Run 'docker system prune -f' on the build agent to clear space.",
        "source": "Slack #devops-support"
    },
    {
        "content": "To skip the unit test stage in the CI pipeline (not recommended for prod), add '[skip tests]' to your commit message.",
        "source": "Confluence: CI/CD Handbook"
    },
    # Frontend & UI
    {
        "content": "The internal 'Amex-Design-System' (ADS) version 5.0 is now live. It includes the new accessible color palette. Update your 'package.json' to '@amex/ads-web: ^5.0.0'.",
        "source": "Slack #frontend-guild"
    },
    # General Troubleshooting
    {
        "content": "If you cannot access the 'Internal-GitHub' via SSH, ensure your public key is added to your profile and that you are using the VPN.",
        "source": "Slack #it-helpdesk"
    },
    {
        "content": "The 'Dev-Portal' goes down for maintenance every Sunday at 2:00 AM EST for approximately 30 minutes.",
        "source": "Slack #announcements"
    }
]

# 2. Prepare documents
documents = [Document(page_content=d["content"], metadata={"source": d["source"]}) for d in sample_data]

# 3. Initialize Embeddings and Vector DB
print("Initializing local embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. Save to local disk
persist_directory = "chroma_db"
print(f"Adding {len(documents)} documents to the vector database in {persist_directory}...")
vectordb = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=persist_directory
)

print("Knowledge ingestion complete! You now have a much larger brain.")
