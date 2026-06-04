SYNTHESIZE_PROMPT = """You are an AI assistant helping with LMS (Learning Management System) questions.

Context information:
---------------------
{context_str}
---------------------

Question: {query_str}

Based on the context above, provide a detailed answer to the question. If the context doesn't contain enough information, keep the answer concise and relevant. Answer in vietnamese.

Answer:"""

COMPACT_AND_REFINE_PROMPT = """You are an AI assistant helping with LMS questions.

Original Question: {query_str}

Existing Answer: {existing_answer}

Additional Context:
---------------------
{context_msg}
---------------------

Using the additional context above, refine and improve the existing answer. If the new context is not helpful, keep the original answer. Answer in vietnamese.

Refined Answer:"""

QUESTION_GENERATION_PROMPT = """You are an AI assistant that supports questions about LMS (Learning Management System).

Context information:
---------------------
{context_str}
---------------------

Question: {query_str}

Based on the above context, break the question into smaller questions with context. The question is in Vietnamese.

Question: """
