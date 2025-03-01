from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Check if the API key is set correctly
if not api_key:
    raise ValueError("API key is missing. Please check your .env file and ensure the OPENAI_API_KEY is correctly configured.")
else:
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

def modelloading(prompt, model="gpt-3.5-turbo"):
    """
    Sends a prompt to OpenAI's API and returns the response.
    Enforces strict formatting to return only the C++ translation.
    """
    try:
        # System prompt for context
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
        )

        # Request completion using OpenAI API
        completion = client.chat.completions.create(
            model=model,  # Select the model (e.g., gpt-3.5-turbo, gpt-4)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,  # Token limit (can adjust as needed)
            temperature=0.7,  # Adjust creativity
        )

        # Return the generated code (response text)
        return completion.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error: {str(e)}"