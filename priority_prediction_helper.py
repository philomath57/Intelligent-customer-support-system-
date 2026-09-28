from sklearn.ensemble import RandomForestClassifier
import spacy
import torch

nlp = spacy.load('en_core_web_sm')

def text_preprocessing(text):

        text = text.lower()
        doc = nlp(text)

        filtered_text = []

        for token in doc:
            filtered_text.append(token.lemma_)

        return ' '.join(filtered_text)

class make_priority_prediction:
    def __init__(self):
        pass

    def priority_prediction(self, text, model_pipeline):

        preprocessed_text = text_preprocessing(text)

        prediction = model_pipeline.predict([preprocessed_text])

        class_names = {"Low": 0, "Medium": 1, "High": 2}

        predicted_class = [k for k, v in class_names.items() if v == prediction.item()]

        return predicted_class

    def queue_prediction(self, text, tokenizer, bert_model):
        input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
         
        with torch.no_grad():
            logits = bert_model(**input).logits
            
        prediction = torch.argmax(logits, dim=1)

        class_names = {'Technical Support': 0, 'Product Support': 1, 'Customer Service': 2,
                          'IT Support': 3, 'Billing and Payments': 4, 'Returns and Exchanges': 5,
                          'Service Outages and Maintenance': 6, 'Sales and Pre-Sales':7, 'Human Resources': 8, 'General Inquiry': 9}

        prediction_label = [k for k, v in class_names.items() if v == prediction.item()]

        return prediction_label

    
    
        





    





