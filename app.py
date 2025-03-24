import os
import json
from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Check for API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error("OpenAI API key is not set!")
    st.stop()

# Load the knowledge base from a JSON file
with open('knowledge_base.json', 'r', encoding='utf-8') as f:
    KNOWLEDGE_BASE = json.load(f)

# Define the system prompt to set the bot's persona and guidelines
SYSTEM_PROMPT = """
 
You are *Zerina Zinger*, a Serbian interior design educator with a warm, casual, and encouraging tone. Your role is to:  
1. *Answer all questions* using the FAQ database *or* respond gracefully if unmatched.  
2. *Mirror Zerina’s Balkan Serbian style* (colloquial phrases, humor, emojis).  
3. *Guide prospects to the newsletter signup* using the exact scripted flow.  

---

### *Workflow & Rules*  
1. *Strict Script Adherence*:  
   - Follow the *Instagram chat structure* step-by-step. No deviations.  
   - Example:  
     - *User: *“Zdravo, vidjela sam vaš oglas o kursu.”  
     - *You: *“Zdravo! Baš mi je drago što si se javila. 🥳 Šta te zanima? Imaš li nešto konkretno na umu?”  

2. *Database-Driven Responses*:  
   - *Priority: Pull answers **verbatim* from the FAQ database. Add Zerina’s flair (emojis, phrases).  
     - Database Answer: “Plaćanje je jednokratno.”  
     - Your Response: “Za sad je jednokratno, ali ako ti je frka sa budžetom, piši mi—možda smislimo nešto! 💸”  

3. *Handling Unmatched Questions*:  
   - *Never ignore a query*. Respond in Zerina’s tone and redirect to the script:  
     - “Dobar pitanje! Trenutno nemam info, ali proveriću sa Zerinom i javiću ti. 😊 A da, šta još te zanima o kursu?”  
   - *Log the question* but keep the conversation flowing toward signup.  

4. *Tone & Style*:  
   - *Phrases: *“bre”, “ma daj”, “nema žurbe”, “znaš kako je”.  
   - *Emojis*: 🌟✨ (enthusiasm), 😉 (reassurance), 🛋️ (design topics).  
   - *Sentence Structure*: Short, punchy, conversational.  
     - “Snimljene su lekcije. Gledaš kad hoćeš, koliko hoćeš. Nema žurbe! 😊”  

5. *Closing Sequence*:  
   - After answering questions (matched or unmatched), deliver the *exact closing message*:  
       
     “Početni kurs traje 30 dana. Nakon toga imaš pristup lekcijama 1 godinu. Ova ponuda traje do 8.4.2025. Unutar mjesec dana imaš moju mentorsku podršku. Komuniciramo preko email-a i WhatsApp...”  
       
   - End with the newsletter prompt:  
       
     “Sviđa ti se ovo što čuješ? Imam newsletter s još savjeta... Hoćeš link? 📩”  
       

6. *Follow-Up*:  
   - If no reply for 30 minutes:  
       
     “Hej, samo da provjerim—imaš li još pitanja? Tu sam kad god treba! 💬”  
       

---


"""

# Prepare few-shot examples from the FAQ (limiting to first 5 examples for prompt size)
few_shot_examples = [
    {
        "input": faq["question"],
        "output": faq["answer"]
    } for faq in KNOWLEDGE_BASE['faqs'][:5]
]

# Create a few-shot prompt template
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=few_shot_examples
)

# Create the full prompt template including the system prompt and the few-shot examples
prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    few_shot_prompt,
    ("human", "{query}")
])

# Define the state of the graph with a typed dictionary
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    knowledge_base: dict

# Initialize the language model (currently using GPT-4o)
# Initialize the language model (using gpt4omini now)
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# Create a processing chain by combining the prompt template and the model
chain = prompt_template | model

# Fallback function to find a matching FAQ using simple text matching
def find_matching_faq(question, faqs):
    for faq in faqs:
        if question.lower() in faq['question'].lower():
            return faq['answer']
    return None

# Function to process messages using the few-shot chain
def process_message(state: AgentState):
    messages = state['messages']
    latest_message = messages[-1].content if messages else ""
    
    try:
        # Generate a response using the chain
        response = chain.invoke({"query": latest_message})
        
        # If the response is too generic, attempt a fallback FAQ match
        if len(response.content.strip()) < 20:
            faq_answer = find_matching_faq(latest_message, KNOWLEDGE_BASE['faqs'])
            if faq_answer:
                response_text = f"{faq_answer} 😊🛋️"
            else:
                response_text = f"{response.content} 😊🛋️"
        else:
            response_text = f"{response.content} 😊🛋️"
        
        ai_message = AIMessage(content=response_text)
        return {"messages": [ai_message]}
    
    except Exception as e:
        # In case of an error, return a fallback error message
        error_message = AIMessage(content=f"Ups, nešto nije u redu. Molim te ponovi pitanje. 🤷‍♀️ Detalji: {str(e)}")
        return {"messages": [error_message]}

# Function to decide if the conversation should end
def should_end(state: AgentState):
    messages = state['messages']
    # For example, end the conversation if more than 5 messages have been exchanged
    if len(messages) > 5:
        return END
    return "process"

# Build the bot using a state graph to manage the conversation flow
def create_bot(knowledge_base):
    graph_builder = StateGraph(AgentState)
    
    # Add node for processing messages
    graph_builder.add_node("process", process_message)
    
    # Add edges: start from START, and conditionally loop or end based on should_end
    graph_builder.add_edge(START, "process")
    graph_builder.add_conditional_edges("process", should_end, 
        {
            "process": "process",
            END: END
        }
    )
    
    # Use a memory saver to persist state if needed
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)
    return graph

# Main interactive loop for the conversation
def main():
    bot = create_bot(KNOWLEDGE_BASE)
    
    # Display the system prompt and initial greeting
    print("🌟 Zerina Zinger's Interior Design Bot 🛋️")
    print(SYSTEM_PROMPT)
    
    # Initialize conversation state with the system prompt as the first message
    state: AgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "knowledge_base": KNOWLEDGE_BASE
    }
    
    while True:
        try:
            user_input = input("\nVi: ").strip()
            # End conversation if user types a goodbye command
            if user_input.lower() in ["doviđenja", "cao", "ćao"]:
                print("AI: Doviđenja! Hvala što ste koristili bot.")
                break
        except KeyboardInterrupt:
            print("\nPrekinuli ste. Hvala na razgovoru!")
            break
        
        # Add the user's message to the conversation state
        state["messages"].append(HumanMessage(content=user_input))
        
        # Process the conversation state to generate a response
        result = process_message(state)
        ai_messages = result.get("messages", [])
        
        # Append and display the AI messages
        for msg in ai_messages:
            state["messages"].append(msg)
            print("AI:", msg.content)
        
        # Optionally, check if the conversation should end based on the state
        if should_end(state) == END:
            print("AI: Hvala na razgovoru, doviđenja!")
            break

if __name__ == "__main__":
    main()
