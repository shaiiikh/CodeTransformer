import os
import asyncio
import openai
from dotenv import load_dotenv
import streamlit as st

# Fix for the "no running event loop" error
# Place this at the very beginning of your script
if not asyncio.get_event_loop().is_running():
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv()

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    # Fallback for Streamlit deployment where .env might not be available
    # You'll need to set this in Streamlit's secrets management
    api_key = st.secrets.get("OPENAI_API_KEY", "")

client = openai.OpenAI(api_key=api_key)

def modelloading(prompt, model="gpt-3.5-turbo"):
    """
    Sends a prompt to OpenAI's API and returns the response.
    Enforces strict formatting to return only the C++ translation.
    """
    try:
        system_prompt = (
            "You are a specialized AI that translates strictly between Pseudocode and C++. "
            # Rest of your system prompt...
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# Rest of your Streamlit app code
# ...

# Example Streamlit UI
st.title("Pseudocode to C++ Translator")
user_input = st.text_area("Enter pseudocode:", height=200)
if st.button("Translate"):
    if user_input:
        with st.spinner("Translating..."):
            result = modelloading(user_input)
        st.code(result, language="cpp")
    else:
        st.warning("Please enter some pseudocode to translate.")