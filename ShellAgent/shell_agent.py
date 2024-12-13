import boto3
br = boto3.client('bedrock-runtime')
import json
import time
import random
from tenacity import retry, stop_after_attempt, wait_random


from langchain.agents import AgentType,initialize_agent
from langchain_community.tools import ShellTool
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_aws import BedrockLLM, ChatBedrockConverse, ChatBedrock

from boto3 import client
from botocore.config import Config

config = Config(
    read_timeout=1000,
    retries = {
      'max_attempts': 25,
      'mode': 'standard'
   })

client = client(service_name='bedrock-runtime', 
                      region_name='us-east-1',
                      config=config)



llm =ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0.001, "max_tokens":4000},
    client = client
)


shell_tool=ShellTool()
agent=initialize_agent([shell_tool],
                       llm,
                       agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                       verbose=True,
                       handle_parsing_errors=True,
                       max_iterations=25,
                       return_intermediate_steps=True)

agent.agent.llm_chain.prompt.messages[0].prompt.template

agent.agent.llm_chain.prompt.messages[0].prompt.template = agent.agent.llm_chain.prompt.messages[0].prompt.template.replace("Thought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin! Reminder to always use the exact characters `Final Answer` when responding.'",
                                                                                                                            "Thought: Let me try a new search strategy (restart iterations if required)\nYOU MUST restart if you don't find an answer. Use regex and multiple pattern matches like 4b) below. \nFinal Answer: Make sure you have deeply searched through with different strategies and then use the exact characters `Final Answer` before responding. Do NOT provide your own answers, and always use the search context to answer questions. If the search does not have the information you are looking for, or only has partial information, try again or do a deeper search based on the previous results! Remember that the search should be comprehensive!")


agent.agent.llm_chain.prompt.messages[0].prompt.template = agent.agent.llm_chain.prompt.messages[0].prompt.template.replace("terminal: Run shell commands on this Linux machine.","""

terminal: Run shell commands on this Linux machine to search information in the "files/" folder. The commands to use are:


```pdfmetadata.sh```
--------------------
YOU MUST first print details of pdf files in the files/ folder (ALWAYS start with this without any changes)
# sh pdfmetadata.sh

This gives you file level metadata that is useful to narrow down the search. Then use rga or pdfgrep. Action input must start with rga or pdfgrep and contain the full command.:

```rga```
---------
A command line tool to search through files via keyword searches and reges patterns. All files relvent to this task are in the files/ folder.

- To find a search term in specific file (use regex pattern)
rga 'searchterm\w*' ./files/filename.pdf

- To search with multiple keyword matches across multiple files:
rga 'keyword1|keyword2|keyword3' ./files/

- Use -i for case insensitive search.


```pdfgrep```
-------------
Another commandline tool specifically for search with PDFs. Useful for special cases when:

- search in a folder with pdfs, across a specific page range for one or more keywords (-i is case insensitive, -n includes page numbers in output, -r is recursive search in files folder, and -P is perl compatible regex). With pdfgrep you must include '(' ')' brackets for the pattern
pdfgrep -inrP --page-range 1-4 '(keyword1|keyword2)' ./files/

- Search all .pdf files whose names begin with foo recursively in the current directory:
pdfgrep -r --include "foo*.pdf" pattern

Other tips:
***********
- If a complex query fails, try a series of simpler queries instead. 
- ALWAYS try to return larger context with -C 5 with both rga nad pdfgrep to get 2 or more lines around the returned keyword match.
- remember that you MUST do 'sh pdfmetadata.sh' first to understand what files you are dealing with and then continue your search.
- your action should always just be "terminal" and action input is the full command you want to run in the terminal 
***********
""")




input_data = json.load(open('./data2/rag_dataset.json','r'))

skipped_questions = []
failed_questions = []

@retry(stop=stop_after_attempt(3), wait=wait_random(min=5, max=10))
def invoke_agent(agent, query):
    return agent.invoke({"input": query})

with open('./data2/output.jsonl', 'w') as outfile:
    for example in input_data['examples']:
        try:
            time.sleep(10)
            response = invoke_agent(agent, example['query'])

            json.dump({
                "output": response['output'],
                "observations": [s[1] for s in response['intermediate_steps']],
                "action_inputs": [s[0].tool_input for s in response['intermediate_steps']],
                "thoughts": [s[0].log for s in response['intermediate_steps']],
            }, outfile)
            outfile.write('\n')

            print("---------------------------")
            print(example['query'])
            print(response['output'])
            print("---------------------------")

        except Exception as e:
            print(f"Error processing question: {example['query']}")
            print(f"Error message: {str(e)}")
            failed_questions.append(example['query'])
            skipped_questions.append(example['query'])
            continue

# Retry skipped questions
for question in skipped_questions:
    try:
        time.sleep(10)
        response = invoke_agent(agent, question)

        json.dump({
            "output": response['output'],
            "observations": [s[1] for s in response['intermediate_steps']],
            "action_inputs": [s[0].tool_input for s in response['intermediate_steps']],
            "thoughts": [s[0].log for s in response['intermediate_steps']],
        }, outfile)
        outfile.write('\n')

        print("---------------------------")
        print(question)
        print(response['output'])
        print("---------------------------")

    except Exception as e:
        print(f"Error processing skipped question: {question}")
        print(f"Error message: {str(e)}")
        failed_questions.append(question)

print("Questions that failed after retries:")
for question in failed_questions:
    print(question)
