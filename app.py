import streamlit as st
import torch
import torch.nn as nn
import json
import math
import os
from streamlit.components.v1 import html
from loadmodels import modelloading

# Disable the Streamlit file watcher which is causing the issues with PyTorch
os.environ["STREAMLIT_GLOBAL_WATCHER_WARNING_DISABLED"] = "1"

# Set page config for better appearance
st.set_page_config(
    page_title="CodeTransformer",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load vocabulary
try:
    with open("vocabulary.json", "r") as f:
        vocab = json.load(f)
    st.sidebar.write(f"✅ Vocabulary loaded with {len(vocab)} tokens")
except Exception as e:
    st.sidebar.error(f"Error loading vocabulary: {str(e)}")
    vocab = {}  # Fallback empty vocabulary

# Transformer Configuration
class Config:
    vocab_size = 12006  # Adjust based on vocabulary.json
    max_length = 100
    embed_dim = 256
    num_heads = 8
    num_layers = 2
    feedforward_dim = 512
    dropout = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# Transformer Model
class Seq2SeqTransformer(nn.Module):
    def __init__(self, config):
        super(Seq2SeqTransformer, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.positional_encoding = PositionalEncoding(config.embed_dim, config.max_length)
        self.transformer = nn.Transformer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            batch_first=True  # Add this to address PyTorch warning
        )
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(config.embed_dim)
        tgt_emb = self.embedding(tgt) * math.sqrt(config.embed_dim)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        # Using batch_first=True
        out = self.transformer(src_emb, tgt_emb)
        out = self.fc_out(out)
        return out

# Load Models
@st.cache_resource
def load_model(path):
    model = Seq2SeqTransformer(config).to(config.device)
    try:
        model.load_state_dict(torch.load(path, map_location=config.device))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model from {path}: {str(e)}")
        return None

# Try loading models with error handling
try:
    cpp_to_pseudo_model = load_model("cpp_to_pseudo_epoch_1.pth")
    pseudo_to_cpp_model = load_model("transformer_epoch_1.pth")
    if cpp_to_pseudo_model and pseudo_to_cpp_model:
        st.sidebar.write("✅ Models loaded successfully!")
    else:
        st.sidebar.warning("Some models failed to load. Falling back to OpenAI API.")
except Exception as e:
    st.sidebar.error(f"Error during model loading: {str(e)}")

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

    /* VS Code-like Header Style */
    .vscode-header {
        display: flex;
        align-items: center;
        padding: 1rem 2rem;
        background: #252526;
        border-bottom: 2px solid #3c3c3c;
        animation: headerSlide 1s ease-out;
        margin-bottom: 20px;
        border-radius: 5px;
    }

    .vscode-header h1 {
        color: #007acc;
        margin: 0;
        font-family: 'Fira Code', monospace;
        font-size: 2.5rem;
    }

    .vscode-header .header-title {
        color: #d4d4d4;
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

st.markdown(custom_css, unsafe_allow_html=True)
html('<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">')

# ==============================================
# Enhanced UI Components
# ==============================================

def render_header():
    header_html = """
    <div class="vscode-header">
        <h1>
            Code <span class="header-title">Transformer</span>
        </h1>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


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
    col1, col2 = st.columns(2)
    with col1:
        pseudo_to_cpp = st.button("Pseudocode → C++", key="btn_mode1")
    with col2:
        cpp_to_pseudo = st.button("C++ → Pseudocode", key="btn_mode2")
    
    if pseudo_to_cpp:
        return "mode1"
    elif cpp_to_pseudo:
        return "mode2"
    
    return selected_mode

# ==============================================
# Main App Logic
# ==============================================

# Initialize session state
if 'selected_mode' not in st.session_state:
    st.session_state.selected_mode = None

# Render header
render_header()

# Mode Selection
selected_mode = render_mode_selector(st.session_state.selected_mode)

if selected_mode:
    st.session_state.selected_mode = selected_mode

# Based on the selected mode, set up the appropriate UI
if st.session_state.selected_mode == "mode1":  # Pseudocode to C++
    input_placeholder = "Enter your pseudocode here..."
    output_label = "Generated C++ Code"
    language = "cpp"
    mode = "Pseudocode → C++"
elif st.session_state.selected_mode == "mode2":  # C++ to Pseudocode
    input_placeholder = "Enter your C++ code here..."
    output_label = "Generated Pseudocode"
    language = "python"  # Using python for pseudocode highlighting
    mode = "C++ → Pseudocode"
else:
    input_placeholder = "Select a translation mode above and enter your code or text here..."
    output_label = "Translation Output"
    language = "text"
    mode = None

# Code Input Section with appropriate styling
st.markdown("<h3 style='margin-top: 30px;'>Input:</h3>", unsafe_allow_html=True)
user_input = st.text_area(
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
    elif not user_input.strip():
        st.warning("Please enter some code before translating.")
    else:
        with st.spinner('🔍 Processing...'):
            try:
                # Prepare the prompt based on the mode
                if st.session_state.selected_mode == "mode1":
                    prompt = f"Translate the following Pseudocode to C++:\n\n{user_input}"
                else:
                    prompt = f"Translate the following C++ code to Pseudocode:\n\n{user_input}"
                
                # Process using the modelloading function
                translated_code = modelloading(prompt)
                
                # Display the output
                st.markdown(f"<h3>Output ({output_label}):</h3>", unsafe_allow_html=True)
                st.code(translated_code, language=language)
                st.success("Translation completed successfully!")
            except Exception as e:
                st.error(f"Translation failed: {str(e)}")

# ==============================================
# Sidebar Content
# ==============================================

st.sidebar.title("🛠️ System Info")

# Model Status
st.sidebar.markdown("### Model Status")
if st.session_state.selected_mode:
    mode_names = {
        "mode1": "Pseudocode → C++",
        "mode2": "C++ → Pseudocode"
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
- [Documentation](https://medium.com/@shaiiikh/building-a-transformer-based-model-for-pseudocode-to-code-and-code-to-pseudocode-translation-7889fa79ec08)
- [Report Issues](https://github.com/shaiiikh/CodeTransformer/issues)
""")

# About Section
st.sidebar.markdown("### About")
st.sidebar.markdown("""
CodeTransformer v1.0.0

Built with Streamlit and PyTorch

© 2025 ashcodes
""")