import openai
from dotenv import load_dotenv
import os
import streamlit as st  # If you're using Streamlit

# Load environment variables
load_dotenv()

# Set the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is correctly set
if not openai.api_key:
    st.error("API Key is not set. Please check your .env file and ensure the OPENAI_API_KEY is correctly configured.")

def modelloading(prompt, model="gpt-3.5-turbo"):
    """
    Sends a prompt to OpenAI's API and returns the response.
    Enforces strict formatting to return only the C++ translation.
    """
    try:
        system_prompt = (
            "You are a specialized AI that translates strictly between Pseudocode and C++. "
            "Do NOT include explanations, headers, or any additional text. "
            "Only output the translated code, exactly matching the dataset structure. "
            "Here are examples of correct translations:\n\n"

            "Pseudocode: in the function gcd(a,b=integers)\n"
            "C++: int gcd(int a, int b) {\n"

            "Pseudocode: if b=1 return a, else call function gcd(b, a%b)\n"
            "C++: return !b ? a : gcd(b, a % b);\n"
            "    }\n\n"

            "Pseudocode: n , nn, ans = integers with ans =0\n"
            "C++: int n, nn, ans = 0;\n\n"

            "Pseudocode: Read n\n"
            "C++: cin >> n;\n\n"

            "Pseudocode: for i=2 to n-1 execute\n"
            "C++: for (int i = 2; i <= n - 1; ++i) {\n\n"

            "Pseudocode: set nn to n\n"
            "C++: nn = n;\n\n"

            "Pseudocode: while nn is not equal to 0, set ans to ans + nn%i, and also set nn= nn/i\n"
            "C++: while (nn) ans += nn % i, nn /= i;\n    }\n\n"

            "Pseudocode: set o to gcd(ans, n-2)\n"
            "C++: int o = gcd(ans, n - 2);\n\n"

            "Pseudocode: print out ans/o '/' (n-2)/o\n"
            "C++: cout << ans / o << '/' << (n - 2) / o << '\\n';\n\n"

            "Pseudocode: FOR i FROM 0 TO 4 DO\n"
            "C++: for(int i = 0; i < 4; ++i) {}\n\n"

            "do not write another other than code or pseudocode, like for example (c++: int x=0;) instead just write (int x=0;)" 

            "Always follow this format exactly."

            "if i enter pseudocode instead of c++ when i select c++ to pseudocode do not do anything and vice versa"
        )

        # Make OpenAI API request
        response = openai.ChatCompletion.create(
            model=model,
            messages=[ 
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )

        # Return the translated content
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"
