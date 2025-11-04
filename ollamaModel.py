from ollama import Client


OLLAMA_URL = "http://localhost:11434"
client = Client(
  host= OLLAMA_URL,
  headers={'x-some-header': 'some-value'}
)
models_list = client.list()
print(f"✅ Connect Successfuly on {OLLAMA_URL}.")

structure = """
You are an SQL expert.
You must only output valid SQL queries — without any explanations or extra text.

Database structure:
Table name: Customer
Fields: id, name, family, gender, RegisterCompanyName, RegisterDate

Your task:
When the user asks a question about the data (for example: "show all customers registered after 2022"),
you must generate the correct SQL query based on the above structure.

Always return only the SQL query, nothing else.
User Query :
"""
prompt = "List of first and last names of all female employees registered at Company 'X'."
#prompt = input("Enter your value: ")

model_name = "llama3.1:8b"

#print(structure + prompt)
try:
    response = client.chat(
        model= model_name,
        messages=[
            {
                "role":"user",
                "content": structure + prompt
            }
        ]
    )
    print(response["message"]["content"])

except Exception as e:
    print(e)