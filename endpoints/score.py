import os
import logging
import json
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    """
    global model, tokenizer
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "model"
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    """
    logging.info("Request received")
    input_text = json.loads(raw_data)["text"]

    # td: params for decoder args
    # default args (bart-large-samsum): max_length=62, min_length=11, length_penalty=1.0, num_beams=6, early_stopping=True
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    result = summarizer(input_text)
    logging.info("Request processed")

    return result