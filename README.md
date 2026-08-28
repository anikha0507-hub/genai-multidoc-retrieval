## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
Building a multidocument agent

### DESIGN STEPS:

#### STEP 1:
Idea & Planning – Identify the problem, collect requirements, and prepare the initial design

#### STEP 2:
Design & Development – Create the 3D model/design and develop the required components.

#### STEP 3:
Testing & Finalization – Test the design, make necessary improvements, and prepare the final model for presentation.

### PROGRAM:
```
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()

urls = [
    "https://s172-29-126-125p8888.lab-aws-production.deeplearning.ai/files/accelerating.pdf",
    "https://s172-29-126-125p8888.lab-aws-production.deeplearning.ai/files/Probabilistic_Inference.pdf",
    "https://s172-29-126-125p8888.lab-aws-production.deeplearning.ai/files/algorithm.pdf",
]

papers = [
    "accelerating.pdf",
    "algorithm.pdf",
    "Probabilistic_Inference.pdf",    
]

from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query(
    "Tell me about the important events in the entire paper"
)
response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(response))
```

### OUTPUT:
<img width="1385" height="262" alt="image" src="https://github.com/user-attachments/assets/6e57bd1c-5ff5-4d4b-9bcc-4728c55e03a5" />
<img width="982" height="776" alt="image" src="https://github.com/user-attachments/assets/6ee4ee62-f22d-462d-80b6-f587fadc054a" />


### RESULT:
Therefore,the program to build a multidocument agent has been executed and the output is verified.
