import streamlit as st
from transformers import pipeline, AutoTokenizer

def run_question_answering(question, context):
    model_name = "deepset/roberta-base-squad2"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
    result = nlp({'question': question, 'context': context})
    return result

def main():
    st.title("Question Answering with Streamlit")
    
    # User input for question and context
    context = st.text_area("Enter the context:")
    question = st.text_input("Enter your question:")

    
    if st.button("Get Answer"):
        if question and context:
            st.write("Answer:")
            result = run_question_answering(question, context)
            st.write(result['answer'])
        else:
            st.warning("Please provide both a question and context.")

if __name__ == "__main__":
    main()
