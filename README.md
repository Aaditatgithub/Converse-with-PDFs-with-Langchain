Sure, here's a README template for the code you provided:

---

# Document Intelligence

Document Intelligence is a Streamlit application that provides various functionalities for processing documents, images, and speech input.

## Features

- **Document Processing**: Upload PDF documents to extract text and analyze their content.
- **Image Processing**: Upload images to detect objects and generate image captions.
- **Speech Input**: Use speech input to ask questions and receive responses.

## Setup

1. Clone the repository:

```bash
git clone https://github.com/your-username/document-intelligence.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit application:

```bash
streamlit run app.py
```

## Usage

1. Choose an option from the navigation sidebar (`Document`, `Image`, `Speech`).
2. Follow the instructions for each option to upload files or provide input.
3. Click on the corresponding button to process the input.
4. View the results displayed on the Streamlit app.

## Dependencies

- `streamlit`: 0.90.0
- `dotenv`: 0.19.1
- `speech_recognition`: 3.8.1
- `PyPDF2`: 1.26.0
- `langchain`: <version>
- `htmlTemplates`: <version>
- `gtts`: 2.2.2
- `PIL`: 8.3.1
- `transformers`: 4.11.2
- `torch`: 1.9.0

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

You can replace `<version>` with the actual version numbers of the dependencies used in your project. Additionally, make sure to provide a valid URL to your GitHub repository and update the `LICENSE` section if needed.
