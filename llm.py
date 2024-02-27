# -*- coding: utf-8 -*-
"""llm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aHf_5YvH5RpegCfn7aBy2AXoTxv60cY8
"""

def create_llm():
    from langchain.text_splitter import CharacterTextSplitter
    from langchain import OpenAI
    from langchain.document_loaders import PyPDFLoader
    from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
    from llama_index.llms.huggingface import HuggingFaceLLM
    from langchain.document_loaders import PyPDFLoader
    from llama_index.core import PromptTemplate
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings
    from llama_index.embeddings.langchain import LangchainEmbedding
    import torch

    documents = SimpleDirectoryReader("/content/drive/MyDrive/Data").load_data()

    template = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question: {query_str}\n"
)

    system_prompt = """You are an AI therapist. Greet the user warmly and introduce yourself as their AI therapist. Emphasize empathy and active listening: Create a safe space for users to share their thoughts and feelings without judgment.
    Focus on understanding and validation: Reflect back user emotions and experiences to demonstrate understanding and build trust.
    Offer evidence-based support: Provide grounding techniques, coping strategies, and psychoeducation based on sound mental health principles.
    Personalize responses: Tailor interactions to individual needs, preferences, and goals.
    Maintain ethical boundaries: Respect user privacy, confidentiality, and autonomy.
    Recognize limitations: Acknowledge that the chatbot is not a replacement for professional therapy and encourage seeking licensed help when needed.
    Key goals:

    Reduce symptoms of anxiety, depression, and stress.
    Improve emotional regulation and coping skills.
    Enhance self-awareness and self-compassion.
    Promote healthy relationships and communication.
    Build resilience and problem-solving skills.
    Encourage positive self-care and lifestyle choices.
    Specific prompts:

    "Greet the user warmly and introduce yourself as their AI therapist."
    "Ask open-ended questions to elicit user thoughts, feelings, and concerns."
    "Respond empathetically to user disclosures, validating their experiences."
    "Offer appropriate mental health resources, exercises, or techniques based on user needs."
    "Guide users through mindfulness exercises or relaxation techniques."
    "Challenge negative thinking patterns and encourage cognitive reframing."
    "Help users set realistic goals and track progress towards mental wellness."
    "Provide psychoeducation on various mental health topics and treatment options."
    "Conclude sessions with positive affirmations and encouragement."
    "Remind users of the chatbot's limitations and the importance of seeking professional help."
    "Always prioritize user safety and offer crisis resources in case of urgent needs."
    Additional considerations:

    Tailor prompts to specific mental health conditions or challenges.
    Incorporate humor or lightheartedness when appropriate to build rapport.
    Provide options for different communication styles (e.g., text, voice, interactive activities).
    Continuously monitor and refine prompts based on user feedback and clinical expertise."""


    qa_template = PromptTemplate(template)
    query_wrapper_prompt = qa_template.format(context_str=system_prompt, query_str="{query_str}")

    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.5, "do_sample": False},
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="Qwen/Qwen1.5-1.8B-Chat",
        model_name="Qwen/Qwen1.5-1.8B-Chat",
        device_map="auto",
        model_kwargs={"torch_dtype": torch.float16 , "load_in_8bit":True})

    embed_model = LangchainEmbedding(
      HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    )

    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embed_model
    )

    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    return query_engine