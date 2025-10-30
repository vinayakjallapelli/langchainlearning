# LangChain Basics

## Python LangChain Libraries

| Library                  | Use Case                                         |
|--------------------------|--------------------------------------------------|
| langchain-openai         | Hosts client libraries for OpenAI and ChatOpenAI |
| langchain                | Base package of LangChain                        |
| langchain-community      |                                                  |
| langchain-text-splitters |                                                  |
| langchain-postgres       |                                                  |

## Langchain interface to interact with any LLM Provider
### LLMs
Takes a string prompt as input, send input to model provider, and then returns model prediction output.
```python
model = OpenAI(model="gpt-4o-mini")
print(model.invoke("The sky is "))
```


### Chat Models
Enables back and forth conversation between user and model. Uses SystemMessage and HumanMessage for conversations.
AIMessage and ToolMessage are also supported. https://docs.langchain.com/oss/python/langchain/messages#message-types
```python
model = ChatOpenAI()
humanMessage = HumanMessage("What is the capital of India")
systemMessage = SystemMessage('''You are a helpful assistant that responds with three exclamation marks.''')
```

## Prompt Templates
Prompt templates allows to construct prompts with dynamic inputs. Below example shows a template with two placeholder variables: **context** and **question**.
Values for these variables are passed dynamically.
```python
template = PromptTemplate.from_template("""Answer the question based on the context below. If the question cannot be
 answered using the information provided, answer with "I don't know".
Context: {context}
Question: {question}
Answer: """)
prompt = template.invoke({
    "context":"""The most recent advancements in NLP are being driven by Large Language Models (LLMs). These models
     outperform their smaller counterparts and have become invaluable for developers who are creating applications with
      NLP capabilities. Developers can tap into these models through Hugging Face's `transformers` library, or by
       utilizing OpenAI and Cohere's offerings through the `openai` and `cohere` libraries, respectively.""",
    "question":"Which model providers offer LLMs?"
})
model = OpenAI(model="gpt-4o-mini")
print(model.invoke(prompt))
```

ChatPromptTemplates can be used for building AI Chat Applications. ChatPromptTemplate provides additional context of role to template messages.
```python
template = ChatPromptTemplate.from_messages([
('system', '''Answer the question based on the context below. If the
question cannot be answered using the information provided, answer with
"I don\'t know".'''),
('human', 'Context: {context}'),
('human', 'Question: {question}'),
])
```

## Structured Outputs
You can instruct LLMs to generate output in a specific structure like JSON, CSV, XML etc.
For exmaple, for JSON, you need to define a schema and provide that schema to LLM so that output can be generated in that format.
```python
class AnswerWithJustification(BaseModel):
    answer: str
    '''the answer to user's question'''
    justification: str
    '''justification to the answer'''

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm = llm.with_structured_output(AnswerWithJustification)
print(structured_llm.invoke("""What weighs more, a pound of bricks or a pound of feathers"""))
```

## Runnable Interfaces
There are three ways to invoke LLM: **invoke** | **batch** | **stream**

_invoke_: takes single input, generates single output

_batch_: takes multiple inputs, generates multiple outputs

_stream_: takes single input, generates multiple outputs
```python
completions = model.invoke("Hi there!")

completions = model.batch(['Hi there!', 'Bye!'])

for token in model.stream('Bye!'):
    print(token)

```

## Imperative and Declarative Programming

### Imperative
LangChain offers a decorator **@chain**, which can be used to convert any function into a LangChain Runnable.
```python
template = ChatPromptTemplate.from_messages([
    ('system','You are a helpful assistant'),
    ('human', '{question}')
])

model = ChatOpenAI()

@chain
def chatbot(values):
    prompt = template.invoke(values)
    return model.invoke(prompt)

chatbot.invoke({"question":"Which model providers offer LLMs?"})
```


### Declarative
LangChain also supports declarative way which offers same behaviour as imperative, but with less coding.

Above example can be re-written with declarative synax as below.
```python
chatbot = template | model
chatbot.invoke({"question":"Which model providers offer LLMs?"})
```