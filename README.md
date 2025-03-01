# CodeTransformer

## Overview
CodeTransformer is a powerful tool designed to translate between pseudocode and C++ code seamlessly. It leverages a custom-trained Seq2Seq transformer model to provide accurate and efficient translations, making it an invaluable resource for students, educators, and developers.

## Features
- **Pseudocode to C++**: Convert algorithmic pseudocode into executable C++ code.
- **C++ to Pseudocode**: Translate C++ code into readable pseudocode for better understanding and documentation.
- **User-Friendly Interface**: Intuitive and easy-to-use interface with a modern design inspired by VS Code.
- **Custom-Trained Models**: The models were trained from scratch using a Seq2Seq transformer architecture on a curated dataset.

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch
- Streamlit

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/shaiiikh/CodeTransformer.git
   cd CodeTransformer

2. Install the required dependencies:
        pip install -r requirements.txt

3. Run the Streamlit app:
        streamlit run app.py



## Usage
1. Select Translation Mode:
- Choose between "Pseudocode → C++" or "C++ → Pseudocode" using the mode selector.

2. Input Your Code:
- Enter your pseudocode or C++ code in the provided text area.

3. Translate:
- Click the "Translate" button to generate the corresponding output.

4. View Results:
-The translated code will be displayed in the output section with syntax highlighting.


## File Structure

CodeTransformer/
├── .ipynb_checkpoints/
├── __pycache__/
├── checkpoints/
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── cpp_to_pseudo_epoch_1.pth
├── loadmodels.py
├── requirements.txt
├── transformer.ipynb
├── transformer_epoch_1.pth
└── vocabulary.json


## Models
The project includes custom-trained models:

- cpp_to_pseudo_epoch_1.pth: Model for translating C++ to pseudocode, trained using a Seq2Seq transformer.
- transformer_epoch_1.pth: Model for translating pseudocode to C++, trained using a Seq2Seq transformer.

These models were trained on a curated dataset to ensure high-quality translations.


## Training Details
1. The Seq2Seq transformer model was trained from scratch using the following steps:
2. Dataset Preparation: A dataset of pseudocode and corresponding C++ code was curated and preprocessed.
3. Model Architecture: A transformer-based Seq2Seq model was implemented using PyTorch.
4. Training: The model was trained for multiple epochs with a custom loss function and optimizer.
5. Evaluation: The model's performance was evaluated on a validation set to ensure accuracy.

For more details on the training process, refer to the transformer.ipynb notebook.


## Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4.Submit a pull request.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Inspired by the need for better tools in educational settings.
- Built using PyTorch and Streamlit.

## Contact
For any questions or feedback, please open an issue on GitHub or contact the maintainers directly. 
