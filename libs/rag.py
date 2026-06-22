BASE_PROMPT = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""

USER_PROMPT_TEMPLATE = """
Question: {question}
Context: {context}
"""

MODEL_COSTS = {
    "claude-haiku-4-5": {
        "input": 1 / 1_000_000,
        "output": 5 / 1_000_000
    },
}

class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        base_prompt=BASE_PROMPT,
        user_prompt_template=USER_PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="claude-haiku-4-5"
    ):
        self.index = index
        self.llm_client = llm_client
        self.base_prompt= base_prompt
        self.course = course
        self.user_prompt_template = user_prompt_template
        self.model = model
        self.__costs = []

    def search(self, query, num_results=5):
        boost_dict = {"question": 3.0, "section": 0.5}
        filter_dict = {"course": self.course}

        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict,
            filter_dict=filter_dict
        )
    
    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["section"])
            lines.append("Q: " + doc["question"])
            lines.append("A: " + doc["answer"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.user_prompt_template.format(
            question=query, context=context
        )
    
    def call_llm(self, prompt):
        input_messages = [
            {"role": "system", "content": self.base_prompt},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=input_messages,
            max_tokens=500
        )

        self._log_cost(response)
        return response.choices[0].message.content
    
    def _log_cost(self, response) -> None:
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens

        input_cost = MODEL_COSTS[self.model]["input"] * input_tokens
        output_cost = MODEL_COSTS[self.model]["output"] * output_tokens
        total_cost = input_cost + output_cost

        self.__costs.append({
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": total_cost
        })

    def get_total_cost(self) -> float:
        return sum(entry["cost"] for entry in self.__costs)

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.call_llm(prompt)
        return answer