from typing import List, Dict, Optional
import spacy
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import networkx as nx
from collections import defaultdict
import re

# Base chunking class
class BaseChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        raise NotImplementedError
    

class SemanticChunker(BaseChunker):
    def __init__(self, chunk_size: int = 10, chunk_overlap: int = 10):
        super().__init__(chunk_size, chunk_overlap)
        self.nlp = spacy.load('en_core_web_sm')
        
    def chunk_text(self, text: str) -> List[str]:
        document = self.nlp(text)
        
        chunks = []
        current_chunk = []
        current_chunk_length = 0
        
        # Extract sentences
        sentences = list(document.sents)
        
        for i, sentence in enumerate(sentences):
            sentence_text = sentence.text.strip()
            sentence_length = len(sentence_text)
            
            # Skip empty sentences
            if not sentence_text:
                continue
                
            # If adding this sentence would exceed chunk size and we have content
            if current_chunk_length + sentence_length > self.chunk_size and current_chunk:
                
                # Join current chunk and add to chunks
                chunk_text = " ".join(current_chunk)
                if chunk_text.strip():  # Only add non-empty chunks
                    chunks.append(chunk_text)
                
                # Handle overlap
                if i > 0 and self.keep_overlap(sentence, sentences[i-1]):
                    # Keep the last sentence for overlap
                    current_chunk = [sentence_text]
                    current_chunk_length = sentence_length
                else:
                    # Start fresh
                    current_chunk = []
                    current_chunk_length = 0
            else:
                # Add sentence to current chunk
                current_chunk.append(sentence_text)
                current_chunk_length += sentence_length
        
        # Add the last chunk if it exists and is not empty
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if chunk_text.strip():  # Only add non-empty chunks
                chunks.append(chunk_text)
            
        # Filter out any empty chunks that might have slipped through
        chunks = [chunk for chunk in chunks if chunk.strip()]
            
        return chunks
    
        
    def keep_overlap(self, sentence, previous_sentence=None) -> bool:
        """"Use three conditions to determine if sentence should be kept"""
        
        """
        ROOT: Main verb/predicate
        nsubj: Nominal subject
        dobj: Direct object
        det: Determiner
        amod: Adjectival modifier
        prep: Preposition
        pobj: Object of preposition
        """
            
        # Check for NER
        entities = [ent.text for ent in sentence.ents] # NER extractions
        keywords = [tok for tok in sentence if tok.is_alpha] # Dependency parsing (relations)
        
        important_ner = any(ent.label_ in ['PERSON', 'ORG', 'GPE'] for ent in sentence.ents)
        if important_ner:
            return True
        
        # Check for key semantic content
        has_key_dependencies = any(token.dep_ in ['ROOT', 'nsubj', 'dobj'] 
                                 for token in sentence)
        if has_key_dependencies:
            return True
        
        # Check for connection to previous sentence
        if previous_sentence:
            # Check for shared entities
            current_entities = {ent.text.lower() for ent in sentence.ents}
            prev_entities = {ent.text.lower() for ent in previous_sentence.ents}
            if current_entities & prev_entities:  # Check for intersection
                return True
            
            # Check for shared key words
            current_keywords = {token.text.lower() for token in sentence 
                              if token.pos_ in ['NOUN', 'VERB', 'PROPN']}
            prev_keywords = {token.text.lower() for token in previous_sentence 
                           if token.pos_ in ['NOUN', 'VERB', 'PROPN']}
            if current_keywords & prev_keywords:  # Check for intersection
                return True
        
        return False
        
        
if __name__ == "__main__":
    chunker = SemanticChunker()
    chunks = chunker.chunk_text("Artificial Intelligence (AI) is a broad field of computer science that focuses on creating machines capable of performing tasks that typically require human intelligence. These tasks include learning, reasoning, problem-solving, and decision-making. AI encompasses a wide variety of technologies, including machine learning, deep learning, and natural language processing")
    print(chunks)