import os
from os import path
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
from typing import Any, TypedDict
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class PodcastState(TypedDict):
    main_text: BaseMessage
    key_points: BaseMessage
    script_essence: BaseMessage
    enhanced_script: BaseMessage


def _create_chat_model(model, temperature, provider="OpenRouter", api_key=None):
    if provider == "OpenAI":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
    else:  # OpenRouter
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY")
        )



def _load_prompt(file_path, timestamp=None):
    # Get the absolute path to the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    
    if timestamp:
        prompt_history_dir = os.path.join(root_dir, "prompt_history")
        base_filename = os.path.basename(file_path)
        history_file = f"{base_filename}_{timestamp}"
        history_path = os.path.join(prompt_history_dir, history_file)
        
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
    
    # If no timestamp provided or file not found, fall back to the original prompt file
    absolute_path = os.path.join(root_dir, file_path)
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f"Prompt file not found: {absolute_path}")
    
    with open(absolute_path, 'r', encoding='utf-8') as file:
        return file.read().strip()    


class PodcastCreationWorkflow:
    def __init__(self, summarizer_model="openai/gpt-4o-mini", scriptwriter_model="openai/gpt-4o-mini", enhancer_model="openai/gpt-4o-mini", timestamp=None, provider="OpenRouter", api_key=None):
        self.summarizer_model = _create_chat_model(summarizer_model, 0, provider, api_key)
        self.scriptwriter_model = _create_chat_model(scriptwriter_model, 0, provider, api_key)
        self.enhancer_model = _create_chat_model(enhancer_model, 0.7, provider, api_key)
        self.timestamp = timestamp

        self.summarizer_system_prompt = _load_prompt("prompts/summarizer_prompt.txt", self.timestamp)
        self.scriptwriter_system_prompt = _load_prompt("prompts/scriptwriter_prompt.txt", self.timestamp)
        self.enhancer_system_prompt = _load_prompt("prompts/enhancer_prompt.txt", self.timestamp)

    

    def run_processor(self, state: PodcastState, input_key: str, output_key: str, system_prompt: str, model: Any) -> PodcastState:
        content = state[input_key].content

        if not content:
            raise ValueError(f"The {input_key} content is empty.")

        print(f"Processing {input_key} to generate {output_key}...")
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{" + input_key + "}")
        ])
        chain = prompt | model
        response = chain.invoke({input_key: content})
        processed_content = response.content.strip()

        state[output_key] = HumanMessage(content=processed_content)
        return state    

    def run_summarizer(self, state: PodcastState) -> PodcastState:
        return self.run_processor(state, 
                                  "main_text", 
                                  "key_points", 
                                  self.summarizer_system_prompt, 
                                  self.summarizer_model)


    def run_scriptwriter(self, state: PodcastState) -> PodcastState:
        return self.run_processor(state, 
                                  "key_points", 
                                  "script_essence", 
                                  self.scriptwriter_system_prompt, 
                                  self.scriptwriter_model)


    def run_enhancer(self, state: PodcastState) -> PodcastState:
        return  self.run_processor(state, 
                                   "script_essence", 
                                   "enhanced_script", 
                                   self.enhancer_system_prompt, 
                                   self.enhancer_model)


    def create_workflow(self) -> StateGraph:
        workflow = StateGraph(PodcastState)
        workflow.set_entry_point("summarizer")
        workflow.add_node("summarizer", self.run_summarizer)
        workflow.add_node("scriptwriter", self.run_scriptwriter)
        workflow.add_node("enhancer", self.run_enhancer)

        workflow.add_edge("summarizer", "scriptwriter")
        workflow.add_edge("scriptwriter", "enhancer")
        workflow.add_edge("enhancer", END)

        return workflow


class Agent:
    def __init__(self, model, temperature, provider="OpenRouter", api_key=None):
        self.model = _create_chat_model(model, temperature, provider, api_key)

class PersonalityCreatorAgent(Agent):
    def __init__(self, model="openai/gpt-4o-mini", prompt=None, provider="OpenRouter"):
        super().__init__(model, 0.7, provider)
        self.prompt_template = prompt or _load_prompt("prompts/personality_creator_prompt.txt")

    def create_personality(self) -> str:
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | self.model
        print("Generating personality for feedback assessment...")
        response = chain.invoke({})
        personality = response.content.strip()
        return personality

class FeedbackAgent(Agent):
    def __init__(self, model="openai/gpt-4o", prompt=None, provider="OpenRouter"):
        super().__init__(model, 0, provider)
        self.prompt_template = prompt or _load_prompt("prompts/prompt.txt")

    def run_feedback(self, original_text: str, final_product: str, personality: str) -> str:
        if not original_text or not final_product or not personality:
            raise ValueError("Original text, final product, and personality are all required for feedback.")

        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | self.model
        print("Generating feedback on the original text and final product...")
        response = chain.invoke({
            "personality": personality,
            "original_text": original_text,
            "final_product": final_product
        })
        feedback = response.content.strip()
        return feedback

class WeightClippingAgent(Agent):
    def __init__(self, model="gpt-4o-mini", prompt=None, provider="OpenAI"):
        super().__init__(model, 0, provider)
        self.prompt_template = prompt or _load_prompt("prompts/weight_clipper_prompt.txt")

    def clean_prompt(self, system_prompt: str, role: str) -> str:
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | self.model
        response = chain.invoke({"role": role, "system_prompt": system_prompt})
        return response.content.strip()

class EvaluatorAgent(Agent):
    def __init__(self, model="openai/gpt-4o", prompt=None, provider="OpenRouter"):
        super().__init__(model, 0, provider)
        self.prompt_template = prompt or _load_prompt("prompts/evaluator_prompt.txt")

    def evaluate_podcasts(self, original_text: str, podcast1: str, podcast2: str) -> str:
        prompt = ChatPromptTemplate.from_template(self.prompt_template)
        chain = prompt | self.model
        response = chain.invoke({
            "original_text": original_text,
            "podcast1": podcast1,
            "podcast2": podcast2
        })
        return response.content.strip()