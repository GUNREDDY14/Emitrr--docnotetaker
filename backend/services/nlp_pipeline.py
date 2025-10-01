# backend/app/services/nlp_pipeline.py
"""
Orchestration layer: glue together preprocessing -> ner -> summarizer -> sentiment -> soap.
"""
from utils.preprocessing import clean_text, split_sentences
from . import ner_extraction, summarizer, sentiment_intent, soap_generator


def _extract_keywords(text: str) -> list:
    """Keyword extraction using spaCy noun chunks + simple filtering.
    Keeps medically-relevant multi-word expressions.
    """
    try:
        import spacy
        for model in ["en_core_web_md", "en_core_web_sm"]:
            try:
                nlp = spacy.load(model, disable=["textcat"])  # keep ner/pos/parser
                break
            except Exception:
                nlp = None
                continue
        if nlp is None:
            return []
        doc = nlp(text)
        candidates = []
        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip()
            # heuristic filters: length, contains alphabetic, not purely stopwords
            if len(phrase) < 3:
                continue
            if not any(c.isalpha() for c in phrase):
                continue
            if len(phrase.split()) == 1 and phrase.lower() in nlp.Defaults.stop_words:
                continue
            # prefer multi-word medical-like phrases
            if len(phrase.split()) >= 2:
                candidates.append(phrase)
        # de-duplicate while preserving order
        seen = set()
        keywords = []
        for c in candidates:
            key = c.lower()
            if key not in seen:
                seen.add(key)
                keywords.append(c)
        return keywords[:20]
    except Exception:
        return []


def process_transcript(text: str, metadata: dict = None):
    text_clean = clean_text(text)
    sentences = split_sentences(text_clean)

    # Entities
    entities = ner_extraction.extract_entities(text_clean)

    # Summarization
    summary = summarizer.summarize_text(text_clean, entities=entities)

    # Keywords
    keywords = _extract_keywords(text_clean)

    # Sentiment + intent
    sentiment_analysis = sentiment_intent.analyze_patient_dialogue(text_clean)
    sentiment = (sentiment_analysis["Sentiment"], sentiment_analysis["Intent"])

    # SOAP generation with comprehensive NLP inputs
    soap = soap_generator.generate_soap(
        text=text_clean,
        entities=entities,
        summary=summary,
        sentiment_analysis=sentiment_analysis,
        keywords=keywords
    )

    return {
        "entities": entities,
        "summary": summary,
        "keywords": keywords,
        "sentiment": {"session": sentiment[0], "intent": sentiment[1]},
        "soap": soap,
    }
