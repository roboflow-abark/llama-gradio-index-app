# Required imports
import os
import gradio as gr
from llama_index.indices.managed.llama_cloud import LlamaCloudIndex
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Get the OpenAI and Llama Cloud API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
LLAMA_CLOUD_EU_API_KEY = os.getenv("LLAMA_CLOUD_EU_API_KEY")
# Set the base URL for LlamaIndex Cloud (European server)
os.environ["LLAMA_CLOUD_BASE_URL"] = "https://api.cloud.eu.llamaindex.ai"

# Try to connect to the two indexes on Llama Cloud
try:
    # Connect to the first index (resume data)
    index_1 = LlamaCloudIndex(
        name="noman-resume-2025-06-23",  # name of the index
        project_name="Default",         # default project name used in LlamaIndex
        organization_id="f2bb21e6-3869-4fb9-b5eb-8f342e1c2aac",  # your organization ID
        api_key=LLAMA_CLOUD_EU_API_KEY,  # API key for authentication
        base_url="https://api.cloud.eu.llamaindex.ai",  # endpoint
    )
    query_engine_1 = index_1.as_query_engine()  # convert to a queryable engine

    # Connect to the second index (sentiment analysis model data)
    index_2 = LlamaCloudIndex(
        name="sentiment-analysis-SFT-2025-06-23",  # name of the second index
        project_name="Default",
        organization_id="f2bb21e6-3869-4fb9-b5eb-8f342e1c2aac",
    
        api_key=LLAMA_CLOUD_EU_API_KEY,
        base_url="https://api.cloud.eu.llamaindex.ai",
    )
    query_engine_2 = index_2.as_query_engine()  # convert to a queryable engine

    print("✅ Successfully connected to both LlamaIndex indexes.")
except Exception as e:
    print(f"❌ Error connecting to indexes: {e}")
    query_engine_1 = query_engine_2 = None  # fallback to None if connection fails

# Chat function that handles user queries
# Takes user message, chat history, and selected index from dropdown
def chat_function(message, history, selected_index):
    # Check if indexes are connected
    if query_engine_1 is None or query_engine_2 is None:
        return "❌ Index connection issue. Please check server logs."
    try:
        # Based on selected index, query the respective one
        if selected_index == "Noman Resume Index":
            response = query_engine_1.query(message)
        else:
            response = query_engine_2.query(message)
        return str(response)  # Return response as a string
    except Exception as e:
        return f"❌ Error: {str(e)}"  # Return error message in case of failure

# Create the Gradio interface for the chatbot
with gr.Blocks() as demo:
    gr.Markdown("## LlamaIndex Cloud Chat")  # Title
    gr.Markdown("Ask questions about your data stored in LlamaIndex Cloud")  # Subtitle

    # Dropdown for selecting which index to chat with
    index_selector = gr.Dropdown(
        choices=["Noman Resume Index", "Sentiment Analysis Index"],  # options for user
        value="Noman Resume Index",  # default selected
        label="Select Index"  # label shown on UI
    )

    # Chat interface using the chat_function defined above
    gr.ChatInterface(
        fn=chat_function,  # function to run on each message
        chatbot=gr.Chatbot(height=600, label="Chatbot", type="messages"),# chatbot window        
        title="LlamaIndex Cloud Chat",  # chatbot title
        description="Ask questions about your data stored in LlamaIndex Cloud",  # chatbot description
        additional_inputs=[index_selector],  # include dropdown for index selection
        examples=[
            ["What is this index about?"],
            ["Can you summarize the main points?"],
            ["What are the key insights from this data?"]
        ],  # example prompts
        cache_examples=False,  # do not cache examples
        theme="soft",  # visual theme
        type="messages"  # type of interaction
    )

# Run the Gradio app when the script is executed directly
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # allow access from all IPs (required for Render)
        server_port=7860,  # default Gradio port
        auth=[("admin", "llama2025"), ("hassan", "password321")],  # username/password login
        auth_message="🔐 Access restricted to authorized users only."  # login message
    )
