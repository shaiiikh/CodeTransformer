import re
import streamlit as st
import torch
import torch.nn as nn
import json
import math
from streamlit.components.v1 import html

# Set page config for better appearance
st.set_page_config(
    page_title="CodeTransformer",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load vocabulary if file exists, otherwise create empty dict
try:
    with open("vocabulary.json", "r") as f:
        vocab = json.load(f)
except FileNotFoundError:
    vocab = {"source": {}, "target": {}}
    st.sidebar.warning("Vocabulary file not found. Using empty vocabulary.")

# ==============================================
# Custom CSS and Animations
# ==============================================

custom_css = """
<style>
    /* Base VS Code Theme */
    :root {
        --vscode-background: #1e1e1e;
        --vscode-editor: #252526;
        --vscode-accent: #007acc;
        --vscode-text: #d4d4d4;
        --vscode-border: #3c3c3c;
        --vscode-hover: #2a2d2e;
    }

    /* Main Container */
    .stApp {
        background: var(--vscode-background);
        color: var(--vscode-text);
    }

    /* VS Code-like Header */
    .vscode-header {
        display: flex;
        align-items: center;
        padding: 1rem 2rem;
        background: var(--vscode-editor);
        border-bottom: 2px solid var(--vscode-border);
        animation: headerSlide 1s ease-out;
        margin-bottom: 20px;
        border-radius: 5px;
    }

    @keyframes headerSlide {
        from { transform: translateY(-50px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }

    /* Logo animation */
    .logo-container {
        animation: rotate 3s infinite alternate;
        margin-right: 15px;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(10deg); }
    }

    /* Code Editor Styling */
    .code-editor {
        background: var(--vscode-editor);
        border: 2px solid var(--vscode-border);
        border-radius: 8px;
        padding: 1rem;
        position: relative;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .code-editor:hover {
        box-shadow: 0 0 15px rgba(0,122,204,0.3);
    }
    
    /* Syntax Highlighting */
    .cpp-keyword { color: #569cd6; }
    .cpp-function { color: #dcdcaa; }
    .cpp-string { color: #ce9178; }
    .cpp-comment { color: #6a9955; }
    .cpp-number { color: #b5cea8; }
    .cpp-operator { color: #d4d4d4; }

    /* Mode Selector */
    .mode-selector {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .mode-card {
        background: var(--vscode-editor);
        border: 1px solid var(--vscode-border);
        border-radius: 8px;
        padding: 1.5rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        flex: 1;
    }

    .mode-card:hover {
        transform: translateY(-5px);
        border-color: var(--vscode-accent);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }

    .mode-card.selected {
        border: 2px solid var(--vscode-accent);
        background: rgba(0,122,204,0.1);
    }
    
    .mode-card .icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }

    /* Output Section */
    .code-output {
        background: #0d0d0d;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #007acc;
        position: relative;
        animation: outputAppear 0.5s ease-out;
        margin-top: 1.5rem;
    }

    @keyframes outputAppear {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Translate Button */
    .stButton>button {
        background: var(--vscode-accent) !important;
        color: white !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 4px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        width: 100% !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        margin-top: 10px !important;
    }

    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(0,122,204,0.5) !important;
        transform: scale(1.03) !important;
    }
    
    /* Status Metrics */
    .status-metric {
        background: var(--vscode-editor);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--vscode-border);
    }
    
    /* Better Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: var(--vscode-editor);
        border-radius: 8px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: var(--vscode-background);
        border-radius: 6px;
        color: var(--vscode-text);
        font-weight: 400;
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--vscode-accent) !important;
        color: white !important;
    }
</style>
"""

# Logos for different translation modes
cpp_logo = """
<div class="logo-container">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="48" height="48">
  <path d="M14.219 12.969h3.406v1.406h-3.406zM14.219 17.594h3.406V19h-3.406z" fill="#0288d1"/>
  <path d="M16 3C8.832 3 3 8.832 3 16s5.832 13 13 13 13-5.832 13-13S23.168 3 16 3zm0 2c6.086 0 11 4.914 11 11s-4.914 11-11 11S5 22.086 5 16 9.914 5 16 5z" fill="#0288d1"/>
  <path d="M21.906 15.219c.367.367.367 1.094 0 1.406-.367.367-1.094.367-1.406 0-.367-.367-.367-1.094 0-1.406.367-.367 1.094-.367 1.406 0zM18.688 15.219c.367.367.367 1.094 0 1.406-.367.367-1.094.367-1.406 0-.367-.367-.367-1.094 0-1.406.367-.367 1.094-.367 1.406 0zM22.5 20.688l1.313 1.781-2.625 1.969L19.5 24.5l-1.969-2.625-2.625 1.969L14.5 22.5l2.625-1.969L14.5 18.5l1.781-1.313 1.969 2.625 1.969-2.625 1.781 1.313z" fill="#0288d1"/>
</svg>
</div>
"""

pseudo_logo = """
<div class="logo-container">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="48" height="48">
  <rect x="4" y="4" width="24" height="24" rx="2" fill="none" stroke="#9370DB" stroke-width="2"/>
  <line x1="8" y1="10" x2="24" y2="10" stroke="#9370DB" stroke-width="2"/>
  <line x1="8" y1="16" x2="24" y2="16" stroke="#9370DB" stroke-width="2"/>
  <line x1="8" y1="22" x2="16" y2="22" stroke="#9370DB" stroke-width="2"/>
</svg>
</div>
"""

translation_logo = """
<div class="logo-container">
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="48" height="48">
  <path d="M16 3C8.832 3 3 8.832 3 16s5.832 13 13 13 13-5.832 13-13S23.168 3 16 3zm0 2c6.086 0 11 4.914 11 11s-4.914 11-11 11S5 22.086 5 16 9.914 5 16 5z" fill="#4CAF50"/>
  <path d="M10 10 L14 14 L10 18" fill="none" stroke="#4CAF50" stroke-width="2"/>
  <path d="M17 10 L13 14 L17 18" fill="none" stroke="#4CAF50" stroke-width="2"/>
  <line x1="10" y1="22" x2="22" y2="22" stroke="#4CAF50" stroke-width="2"/>
</svg>
</div>
"""

st.markdown(custom_css, unsafe_allow_html=True)
html('<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">')

# ==============================================
# Enhanced UI Components
# ==============================================

def render_header():
    header = """
    <div class="vscode-header" style="display: flex; align-items: center; gap: 10px;">
        <div class="logo-container">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="48" height="48">
                <path d="M14.219 12.969h3.406v1.406h-3.406zM14.219 17.594h3.406V19h-3.406z" fill="#0288d1"/>
                <path d="M16 3C8.832 3 3 8.832 3 16s5.832 13 13 13 13-5.832 13-13S23.168 3 16 3zm0 2c6.086 0 11 4.914 11 11s-4.914 11-11 11S5 22.086 5 16 9.914 5 16 5z" fill="#0288d1"/>
                <path d="M21.906 15.219c.367.367.367 1.094 0 1.406-.367.367-1.094.367-1.406 0-.367-.367-.367-1.094 0-1.406.367-.367 1.094-.367 1.406 0zM18.688 15.219c.367.367.367 1.094 0 1.406-.367.367-1.094.367-1.406 0-.367-.367-.367-1.094 0-1.406.367-.367 1.094-.367 1.406 0zM22.5 20.688l1.313 1.781-2.625 1.969L19.5 24.5l-1.969-2.625-2.625 1.969L14.5 22.5l2.625-1.969L14.5 18.5l1.781-1.313 1.969 2.625 1.969-2.625 1.781 1.313z" fill="#0288d1"/>
            </svg>
        </div>
        <h1 style="color: #007acc; margin: 0; font-family: 'Fira Code', monospace; font-size: 2.5rem;">
            Code<span style="color: #d4d4d4;">Transformer</span>
        </h1>
    </div>
    """
    st.markdown(header, unsafe_allow_html=True)


def render_mode_selector(selected_mode=None):
    html_content = """
    <div class="mode-selector">
        <div class="mode-card" id="mode1" onclick="selectMode('mode1')">
            <div class="icon">📝➡️💻</div>
            <h3>Pseudocode → C++</h3>
            <p>Convert algorithmic pseudocode to C++ implementation</p>
        </div>
        <div class="mode-card" id="mode2" onclick="selectMode('mode2')">
            <div class="icon">💻➡️📝</div>
            <h3>C++ → Pseudocode</h3>
            <p>Translate C++ code to readable pseudocode</p>
     
    </div>

    <script>
    function selectMode(modeId) {
        // Clear all selections
        document.querySelectorAll('.mode-card').forEach(card => {
            card.classList.remove('selected');
        });
        
        // Mark the selected one
        document.getElementById(modeId).classList.add('selected');
        
        // Set the hidden input value
        document.getElementById('selected_mode_input').value = modeId;
        
        // Submit the form
        document.getElementById('mode_form').submit();
    }
    
    // Apply initial selection if set
    document.addEventListener('DOMContentLoaded', function() {
        const initialMode = document.getElementById('selected_mode_input').value;
        if (initialMode) {
            document.getElementById(initialMode).classList.add('selected');
        }
    });
    </script>
    
    <form id="mode_form" method="post">
        <input type="hidden" id="selected_mode_input" name="selected_mode" value="{selected_mode or ''}">
    </form>
    """
    
    st.markdown(html_content, unsafe_allow_html=True)
    
    # Using Streamlit buttons as fallback for the JavaScript functionality
    col1, col2, col3 = st.columns(3)
    with col1:
        pseudo_to_cpp = st.button("Pseudocode → C++", key="btn_mode1")
    with col2:
        cpp_to_pseudo = st.button("C++ → Pseudocode", key="btn_mode2")
    with col3:
        arabic_to_english = st.button("Arabic → English", key="btn_mode3")
    
    if pseudo_to_cpp:
        return "mode1"
    elif cpp_to_pseudo:
        return "mode2"
    elif arabic_to_english:
        return "mode3"
    
    return selected_mode

# ==============================================
# Transformer Model Implementation
# ==============================================

# Placeholder for your Transformer model classes
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, 
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # Embedding layers
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
    
    def forward(self, src, tgt):
        # Embedding and positional encoding
        src_embedded = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_embedded = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        
        # Generate masks
        src_padding_mask = (src == 0).transpose(0, 1)
        tgt_padding_mask = (tgt == 0).transpose(0, 1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        
        # Forward through transformer
        output = self.transformer(
            src=src_embedded, 
            tgt=tgt_embedded,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask
        )
        
        # Project to vocabulary size
        return self.output_layer(output)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ==============================================
# Model Loading Functions
# ==============================================

@st.cache_resource
def load_model(mode):
    """Load the appropriate model based on the selected mode"""
    try:
        if mode == "mode1":  # Pseudocode to C++
            model_path = "models/pseudo_to_cpp_model.pt"
            src_vocab_size = len(vocab.get("pseudo", {}))
            tgt_vocab_size = len(vocab.get("cpp", {}))
        elif mode == "mode2":  # C++ to Pseudocode
            model_path = "models/cpp_to_pseudo_model.pt"
            src_vocab_size = len(vocab.get("cpp", {}))
            tgt_vocab_size = len(vocab.get("pseudo", {}))
        elif mode == "mode3":  # Arabic to English
            model_path = "models/arabic_to_english_model.pt"
            src_vocab_size = len(vocab.get("arabic", {}))
            tgt_vocab_size = len(vocab.get("english", {}))
        else:
            return None

        # Create a placeholder model for demo purposes
        # In a real implementation, you would load the actual trained model
        model = TransformerModel(
            src_vocab_size=max(src_vocab_size, 1000),  # Ensure minimum vocab size
            tgt_vocab_size=max(tgt_vocab_size, 1000)
        )
        
        # Try to load the model weights
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            st.sidebar.success(f"✅ Model loaded successfully from {model_path}")
        except FileNotFoundError:
            st.sidebar.warning(f"⚠️ Model file not found at {model_path}. Using untrained model.")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading model: {str(e)}")
        
        return model
    
    except Exception as e:
        st.sidebar.error(f"❌ Failed to initialize model: {str(e)}")
        return None

# ==============================================
# Tokenization and Inference Functions
# ==============================================

def tokenize_text(text, vocab_dict):
    """Tokenize input text using the provided vocabulary"""
    # Simple whitespace tokenization for demo
    tokens = text.strip().split()
    # Convert tokens to indices
    indices = [vocab_dict.get(token, vocab_dict.get("<unk>", 1)) for token in tokens]
    return indices

def detokenize_text(indices, vocab_dict_reverse):
    """Convert token indices back to text"""
    # Convert indices to tokens
    tokens = [vocab_dict_reverse.get(idx, "<unk>") for idx in indices]
    # Join tokens into text
    text = " ".join(tokens)
    return text

def translate_code(input_text, model, mode):
    """Translate the input text using the selected model"""
    # In a real implementation, you would:
    # 1. Tokenize the input text
    # 2. Convert to tensor
    # 3. Run inference with the model
    # 4. Decode the output
    
    # For demonstration purposes, we'll return placeholder outputs
    if mode == "mode1":  # Pseudocode to C++
        return """#include <iostream>
#include <vector>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    int sum = 0;
    
    for (int num : numbers) {
        sum += num;
    }
    
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}"""
    elif mode == "mode2":  # C++ to Pseudocode
        return """ALGORITHM ComputeSum
INPUT: List of integers 'numbers'
OUTPUT: Sum of all numbers

SET sum = 0
FOR EACH number IN numbers:
    ADD number TO sum
END FOR
PRINT "Sum: " + sum
RETURN sum"""
    elif mode == "mode3":  # Arabic to English
        return "Hello, this is a translation from Arabic to English."
    else:
        return "Please select a translation mode first."

# ==============================================
# Main App Logic
# ==============================================

def main():
    render_header()
    
    # Initialize session state
    if 'selected_mode' not in st.session_state:
        st.session_state.selected_mode = None
        
    # Mode Selection
    selected_mode = render_mode_selector(st.session_state.selected_mode)
    
    if selected_mode:
        st.session_state.selected_mode = selected_mode
    
    # Display different input prompts based on selected mode
    if st.session_state.selected_mode == "mode1":
        input_placeholder = "Enter your pseudocode here..."
        output_label = "Generated C++ Code"
        language = "cpp"
    elif st.session_state.selected_mode == "mode2":
        input_placeholder = "Enter your C++ code here..."
        output_label = "Generated Pseudocode"
        language = "python"  # Using python for pseudocode highlighting
    elif st.session_state.selected_mode == "mode3":
        input_placeholder = "أدخل النص العربي هنا..."  # Enter Arabic text here
        output_label = "English Translation"
        language = "text"
    else:
        input_placeholder = "Select a translation mode above and enter your code or text here..."
        output_label = "Translation Output"
        language = "text"
    
    # Code Input Section with appropriate styling
    st.markdown("<h3 style='margin-top: 30px;'>Input:</h3>", unsafe_allow_html=True)
    input_text = st.text_area(
        label="",
        value="",
        placeholder=input_placeholder,
        height=250,
        key="input_text"
    )
    
    # Translation Button
    if st.button("✨ Translate", key="translate_button"):
        if not st.session_state.selected_mode:
            st.error("Please select a translation mode first!")
        elif not input_text.strip():
            st.warning("Please enter some text to translate!")
        else:
            with st.spinner('🔍 Processing...'):
                # Load the appropriate model
                model = load_model(st.session_state.selected_mode)
                
                if model:
                    # Perform the translation
                    translated_text = translate_code(input_text, model, st.session_state.selected_mode)
                    
                    # Display the output
                    st.markdown(f"<h3>Output ({output_label}):</h3>", unsafe_allow_html=True)
                    st.code(translated_text, language=language)
                    
                    # Display success message
                    st.success("Translation completed successfully!")
                else:
                    st.error("Failed to load the translation model. Please try again later.")
    
    # ==============================================
    # Sidebar Content
    # ==============================================
    
    st.sidebar.title("🛠️ System Info")
    
    # Model Status
    st.sidebar.markdown("### Model Status")
    if st.session_state.selected_mode:
        mode_names = {
            "mode1": "Pseudocode → C++",
            "mode2": "C++ → Pseudocode",
            "mode3": "Arabic → English"
        }
        st.sidebar.info(f"Active Model: {mode_names.get(st.session_state.selected_mode, 'None')}")
    else:
        st.sidebar.info("No model selected")
    
    # System Metrics
    st.sidebar.markdown("### System Metrics")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("CPU Usage", "45%", "2%")
    col2.metric("Memory", "1.2 GB", "-0.1 GB")
    
    # Translation Statistics
    st.sidebar.markdown("### Translation Stats")
    st.sidebar.progress(75, text="Translation Quality")
    
    # Help & Documentation
    st.sidebar.markdown("### Help & Resources")
    st.sidebar.markdown("""
    - [Documentation](https://docs.example.com)
    - [Report Issues](https://github.com/example/issues)
    - [Tutorial Video](https://youtube.com)
    """)
    
    # About Section
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    CodeTransformer v1.0.0
    
    Built with Streamlit and PyTorch
    
    © 2025 ashcodes
    """)

if __name__ == "__main__":
    main()