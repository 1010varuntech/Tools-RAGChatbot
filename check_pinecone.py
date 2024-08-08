import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Read Pinecone API details from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_environment)

# Check if the index exists and connect to it
if pinecone_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=pinecone_index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(pinecone_index_name)

# Function to check namespace content
def check_namespace(namespace):
    try:
        # Fetch metadata about the namespace
        index_info = index.describe_index_stats(namespace=namespace)
        print(f"Index info for namespace '{namespace}':\n", index_info)
        
        # Create a dummy vector for querying
        dummy_vector = np.random.rand(1536).tolist()

        # Query the index to get sample vectors (adjust the top_k as needed)
        query_response = index.query(
            vector=dummy_vector,
            namespace=namespace,
            top_k=10,  # Fetch top 10 vectors for demonstration
            include_metadata=True
        )
        print(f"\nSample vectors from namespace '{namespace}':\n", query_response)
    except Exception as e:
        print(f"Error checking namespace '{namespace}': {str(e)}")

# Define the namespaces to check
namespaces = ["www.linkedin.com", "www.dripify.io"]

# Check each namespace
for ns in namespaces:
    print(f"Checking namespace: {ns}")
    check_namespace(ns)
    print("\n" + "="*50 + "\n")
