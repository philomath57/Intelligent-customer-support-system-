from semantic_retrieval import perform_semantic_retrieval

class perform_rag_operation:
    def __init__(self):
        pass

    def rerank_candidates(self, query, candidates, reranker, top_k=3):

        pairs = [(query, row['body']) for _, row in candidates.iterrows()]
        reranker_score = reranker.predict(pairs)
        candidates = candidates.copy()

        candidates['reranker_score'] = (reranker_score)

        candidates = candidates.sort_values(by='reranker_score', ascending=False).reset_index(drop=True)
        candidates['rank'] = range(1, len(candidates)+1)

        candidates = candidates.head(top_k)

        return candidates

    def build_rag_context(self, retrieved_context):
        context_part = []

        for _, row in retrieved_context.iterrows():

            context = f"""

            Customer_Issue: {row['body']}

            Previous_Answer: {row['answer']}
"""

            context_part.append(context.strip())

        return "\n\n".join(context_part)

    

    def create_rag_prompt(self, query, context):

        prompt = f"""
Answer the customer support question below using the previous support answers.

Customer question:
{query}

Previous support cases:
{context}

Write a concise and helpful customer support response.

Use the previous answers as guidance.
Do not invent information.
Do not mention these previous cases.
If the information is not sufficient, say that the issue needs to be reviewed by a support agent.

Response:
"""
        return prompt

    def generate_response(self, query, context, model, tokenizer):
        
        prompt = self.create_rag_prompt(query, context)
    
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)

        output = model.generate(**inputs, max_new_tokens=400, num_beams=4, early_stopping=True)

        response = tokenizer.decode(output[0], skip_special_tokens=True)

        return response
