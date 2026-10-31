import os
os.environ["HF_HUB_OFFLINE"] = "1"
from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="Athuin/tinyLama-german",
    task="text-generation",
    pipeline_kwargs=dict(
        max_new_tokens=200,
        do_sample=False,
        repetition_penalty=1.03,
        return_full_text=False,
    ),
)

print(llm.invoke("What is RAG? in 200 words"))