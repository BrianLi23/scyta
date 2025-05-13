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
    """Chunks text based on semantic meaning for searching"""
    def __init__(self, chunk_size: int = 10, chunk_overlap: int = 10):
        super().__init__(chunk_size, chunk_overlap)
        self.nlp = spacy.load('en_core_web_sm')
        
        
    def chunk_text(self, text: str) -> List[str]:
        document = self.nlp(text)
        
        chunks = []
        current_chunk = ""
        current_chunk_length = 0
        
        # Extract sentences
        sentences = list(document.sents)
        for sentence_index in range(len(sentences)):
            current_sentence = sentences[sentence_index]
            if current_chunk_length + len(current_sentence) < self.chunk_size:
                current_chunk += " " + current_sentence.text
                current_chunk_length += len(current_sentence.text)
                
            else:
                chunks.append(current_chunk)
                
                # Keep last sentence if semantic information captured (semantic information)
                # Provides us with sliding window effect of semantic chunking, keeping related sentences with one another.
                if sentence_index > 0 and self.keep_overlap(current_sentence, sentences[sentence_index-1]):
                    current_chunk = current_sentence.text
                    current_chunk_length = len(current_sentence.text)
                else:
                    current_chunk = ""
                    current_chunk_length = 0
                    
        if current_chunk:
            chunks.append(current_chunk)
            
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
        
        # # Check for connection to previous context
        # if previous_sentence:
        #     previous_keywords = set(entity for entity in previous_sentence.ents)
        #     keywords = set(keywords)
            
        #     if any(keyword in previous_keywords for keyword in keywords):
        #         return True
            
        # Checks for key semantic content, such as the main verb or nominal subject
        if entities or any(dependency.dep_ in ['ROOT', 'nsubj'] for dependency in keywords):
            return True
        
        return False
        
        
if __name__ == "__main__":
    chunker = SemanticChunker()
    chunker.chunk_text("This is a test sentence. This is another test sentence. This is a third test sentence.")