from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate

client_side = InferenceClient(model = 'openai/gpt-oss-20b')

messages = [
  {"role": "user", "content": "What does finetuning mean"} 
]

response = client_side.chat_completion(messages)

#print(type(response))

#print(response.choices[0].message.content)

template = PromptTemplate.from_template("Extract features from this record: {raw_text}")

formatted = template.invoke({"raw_text": "some example record here"})

print(formatted)