import streamlit as st
import torch
import torch.nn as nn
import json
import math
from loadmodels import modelloading

# Load vocabulary
with open("vocabulary.json", "r") as f:
    vocab = json.load(f)

st.sidebar.write(f"✅ Vocabulary loaded with {len(vocab)} tokens")

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
def load_model(path):
    model = Seq2SeqTransformer(config).to(config.device)
    model.load_state_dict(torch.load(path, map_location=config.device))
    model.eval()
    return model

cpp_to_pseudo_model = load_model("cpp_to_pseudo_epoch_1.pth")
pseudo_to_cpp_model = load_model("transformer_epoch_1.pth")

st.sidebar.write("✅ Models loaded successfully!")

# Streamlit UI
st.title("C++ & Pseudocode Translator")
mode = st.radio("Select Translation Mode", ("C++ → Pseudocode", "Pseudocode → C++"))
user_input = st.text_area("Enter code:")

if st.button("Translate"):
    if user_input.strip():
        if mode == "C++ → Pseudocode":
            prompt = f"Translate the following C++ code to Pseudocode:\n\n{user_input}"
        else:
            prompt = f"Translate the following Pseudocode to C++:\n\n{user_input}"
        
        translated_code = modelloading(prompt)  

        st.subheader("Generated Translation:")
        st.code(translated_code, language="cpp" if mode == "Pseudocode → C++" else "python")
    else:
        st.warning("Please enter some code before translating.")
