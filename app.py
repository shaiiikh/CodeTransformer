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
    st.sidebar.write(f"✅ Vocabulary loaded with {len(vocab)} tokens")
except FileNotFoundError:
    vocab = {"source": {}, "target": {}}
    st.sidebar.warning("Vocabulary file not found. Using empty vocabulary.")

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
    
    /* Mode button styling */
    .mode-button {
        background-color: var(--vscode-editor);
        border: 1px solid var(--vscode-border);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 10px;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .mode-button:hover {
        border-color: var(--vscode-accent);
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .mode-button.selected {
        border: 2px solid var(--vscode-accent);
        background: rgba(0,122,204,0.1);
    }
    
    .mode-icon {
        font-size: 24px;
        margin-bottom: 10px;
    }
</style>
"""

# Include fonts
html('<link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;600&display=swap" rel="stylesheet">')

# ==============================================
# Model Implementation
# ==============================================

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
            dropout=config.dropout
        )
        self.fc_out = nn.Linear(config.embed_dim, config.vocab_size)

    def forward(self, src, tgt):
        src_emb = self.embedding(src) * math.sqrt(config.embed_dim)
        tgt_emb = self.embedding(tgt) * math.sqrt(config.embed_dim)
        src_emb = self.positional_encoding(src_emb)
        tgt_emb = self.positional_encoding(tgt_emb)
        out = self.transformer(src_emb.permute(1, 0, 2), tgt_emb.permute(1, 0, 2))
        out = self.fc_out(out.permute(1, 0, 2))
        return out

# Load Models
@st.cache_resource
def load_model(model_path):
    try:
        model = Seq2SeqTransformer(config).to(config.device)
        model.load_state_dict(torch.load(model_path, map_location=config.device))
        model.eval()
        st.sidebar.success(f"✅ Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        st.sidebar.warning(f"⚠️ Model file not found at {model_path}. Using untrained model.")
        model = Seq2SeqTransformer(config).to(config.device)
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        return None

# Translation Function
def translate(model, input_tokens, vocab, device, max_length=50):
    if model is None:
        return "Model could not be loaded. Please check the model path."
    
    model.eval()
    try:
        input_ids = [vocab.get(token, vocab.get("<unk>", 1)) for token in input_tokens]
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
        output_ids = [vocab.get("<start>", 0)]
        
        for _ in range(max_length):
            output_tensor = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                predictions = model(input_tensor, output_tensor)
            next_token_id = predictions.argmax(dim=-1)[:, -1].item()
            output_ids.append(next_token_id)
            if next_token_id == vocab.get("<end>", 2):
                break
        
        id_to_token = {idx: token for token, idx in vocab.items()}
        return " ".join([id_to_token.get(idx, "<unk>") for idx in output_ids[1:]])
    except Exception as e:
        return f"Translation error: {str(e)}"

# ==============================================
# Enhanced UI Components
# ==============================================

def render_header():
    st.markdown(custom_css, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="vscode-header">
            <h1>Code <span style="color: #007acc;">Transformer</span></h1>
        </div>
        """, 
        unsafe_allow_html=True
    )

# ==============================================
# Main App Logic
# ==============================================

def main():
    # Apply styling and render header
    render_header()
    
    # Initialize session state
    if 'selected_mode' not in st.session_state:
        st.session_state.selected_mode = None
    
    # Mode Selection Section
    st.markdown("## Select Translation Mode")
    
    col1, col2 = st.columns(2)
    
    # Class to style buttons based on selection
    mode1_class = "mode-button selected" if st.session_state.selected_mode == "mode1" else "mode-button"
    mode2_class = "mode-button selected" if st.session_state.selected_mode == "mode2" else "mode-button"
    
    with col1:
        st.markdown(f"""
            <div class="{mode1_class}">
                <div class="mode-icon">📝➡️💻</div>
                <h3>Pseudocode → C++</h3>
                <p>Convert algorithmic pseudocode to C++ implementation</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Select Pseudocode → C++", key="btn_mode1"):
            st.session_state.selected_mode = "mode1"
            st.rerun()
    
    with col2:
        st.markdown(f"""
            <div class="{mode2_class}">
                <div class="mode-icon">💻➡️📝</div>
                <h3>C++ → Pseudocode</h3>
                <p>Translate C++ code to readable pseudocode</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Select C++ → Pseudocode", key="btn_mode2"):
            st.session_state.selected_mode = "mode2"
            st.rerun()
    
    # Display different input prompts based on selected mode
    if st.session_state.selected_mode == "mode1":
        input_placeholder = "Enter your pseudocode here..."
        output_label = "Generated C++ Code"
        language = "cpp"
        model_path = "p2c1.pth"  # Pseudocode to C++
        st.markdown("### 📝 Pseudocode to C++ Translation")
    elif st.session_state.selected_mode == "mode2":
        input_placeholder = "Enter your C++ code here..."
        output_label = "Generated Pseudocode"
        language = "python"  # Using python for pseudocode highlighting
        model_path = "c2p1.pth"  # C++ to Pseudocode
        st.markdown("### 💻 C++ to Pseudocode Translation")
    else:
        input_placeholder = "Select a translation mode above and enter your code or text here..."
        output_label = "Translation Output"
        language = "text"
        model_path = None
    
    # Code Input Section with appropriate styling
    st.markdown("<h3 style='margin-top: 30px;'>Input:</h3>", unsafe_allow_html=True)
    input_text = st.text_area(
    label="Input Code",  # Give it a meaningful label
    placeholder=input_placeholder,
    height=250,
    key="input_text",
    label_visibility="collapsed"  # Hides the label visually while keeping accessibility
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
                if model_path:
                    model = load_model(model_path)
                    
                    if model:
                        # Tokenize input
                        tokens = input_text.strip().split()
                        
                        # Perform the translation
                        translated_text = translate(model, tokens, vocab, config.device)
                        
                        # Display the output
                        st.markdown(f"<h3>Output ({output_label}):</h3>", unsafe_allow_html=True)
                        st.code(translated_text, language=language)
                        
                        # Display success message
                        st.success("Translation completed successfully!")
                    else:
                        st.error("Failed to load the translation model. Please try again later.")
                else:
                    st.error("Please select a translation mode first.")
    
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