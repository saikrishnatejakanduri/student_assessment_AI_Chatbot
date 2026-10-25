STUDENT ASSISTANT CHATBOT — NO OLLAMA VERSION

Files
- app.py: Streamlit chat interface and TF-IDF intent matching
- intents.json: example intent patterns and responses (replace with your own if needed)
- requirements.txt: required Python packages

SETUP (Windows)
1. Extract this ZIP file.
2. Open Command Prompt in the extracted Student_Assistant_Chatbot folder.
3. Install dependencies:
   python -m pip install -r requirements.txt
4. Start the app:
   python -m streamlit run app.py
5. A browser tab should open. If not, open the Local URL shown in Command Prompt (usually http://localhost:8501).

IMPORTANT
- This version does NOT use Ollama, a generative AI model, a paid API, or an API key.
- It uses TF-IDF similarity to match user messages to example patterns in intents.json and returns the corresponding response plus a follow-up prompt.
- It does not generate unrestricted detailed answers like a large language model. Improve results by adding varied, realistic patterns and useful responses to intents.json.
- Keep app.py and intents.json in the same folder.
- Stop the app with Ctrl+C in the Command Prompt where Streamlit is running.
