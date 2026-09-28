import streamlit as st
from priority_prediction_helper import make_priority_prediction
from semantic_retrieval import perform_semantic_retrieval
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from rag import perform_rag_operation
from sentence_transformers import CrossEncoder


st.set_page_config(page_title="Intelligent Customer Support", page_icon="🎧", layout='wide')

st.title("Intelligent Customer Support")



model_pipeline = joblib.load("priority_classification_model/rf_tfidf_pipeline_priority.pkl") 

bert_model_save_dir = './bert_model_for_queue_classification'

queue_tokenizer = AutoTokenizer.from_pretrained(bert_model_save_dir) 
queue_bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_save_dir)

index = faiss.read_index("semantic_search_using_BERT/support_ticket_faiss.index")
metadata_df = pd.read_pickle("semantic_search_using_BERT/retrieval_metadata_df.pkle")
semantic_model = SentenceTransformer("semantic_search_using_BERT/semantic_bert_model")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

generator_model_name = 'google/flan-t5-base'

generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)

user_input = st.text_area("Please enter your query")

prediction_helper = make_priority_prediction()

semantic_retrieval = perform_semantic_retrieval()

rag_operations = perform_rag_operation()


if st.button("Predict and Retrieve Data"):
    priority_prediction_result = prediction_helper.priority_prediction(text=user_input, model_pipeline=model_pipeline)
    st.subheader("Customer Query Priority Level")
    st.success(priority_prediction_result[0])

    queue_prediction_result = prediction_helper.queue_prediction(text=user_input, tokenizer=queue_tokenizer, 
                                                                 bert_model=queue_bert_model)
    st.subheader("Customer Query Type")
    st.success(queue_prediction_result[0])

    st.subheader("Retrieved Results")
    semantic_result = semantic_retrieval.retrieval_function(query=user_input, model=semantic_model, metadata_df=metadata_df,
                                                            index=index, top_k=10)
    st.dataframe(semantic_result[['body', 'similarity', 'rank']])

    rerank_candidates = rag_operations.rerank_candidates(query=user_input, candidates=semantic_result,
                                                         reranker=reranker, top_k=3)

    rag_context = rag_operations.build_rag_context(rerank_candidates)


    model_response = rag_operations.generate_response(query=user_input, context=rag_context, model=generator_model, 
                                                      tokenizer=generator_tokenizer)

    
    st.subheader('LLM Response')
    st.success(model_response)








