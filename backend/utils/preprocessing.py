# backend/app/utils/preprocessing.py
import re


def clean_text(text: str) -> str:
    # simple cleaning: normalize spaces and unify quotes
    text = text.replace("\r\n", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str):
    # naive sentence splitter
    import re
    sents = re.split(r'(?<=[\.\?\!])\s+', text.strip())
    return [s for s in sents if s]
