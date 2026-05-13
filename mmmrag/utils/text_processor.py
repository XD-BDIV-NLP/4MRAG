#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
text processing utilityModule
ProvidesText、、Functionality
"""

import re
import jieba
import jieba.analyse
import string
from typing import List, Dict, Any, Optional, Union
from collections import Counter
from .logger import get_logger

class TextProcessor:
    """
    text processing utilityClass
    Providescomprehensivetext processingFunctionality
    """
    
    # Class, alreadyoutputInitializeInfo
    _stopwords_initialized = False
    
    def __init__(self, **kwargs):
        """
        Initializetext processor
        
        Args:
            language: languagesetting ('zh'  'en')
            stopwords_path: stopwordsFilePath
            **kwargs: Parameter
        """
        self.language = kwargs.get("language", "zh")
        self.stopwords_path = kwargs.get("stopwords_path", None)
        self.stopwords = set()
        
        # Loadstopwords
        self._load_stopwords()
        
        # Initializejieba(tokenize)
        if self.language == "zh":
            try:
                jieba.initialize()
            except Exception as e:
                print(f"Initialize jieba failed: {e}")
    
    def _load_stopwords(self):
        """
        Loadstopwords list
        """
        # Defaultstopwords list
        if self.language == "zh":
            default_stopwords = {
                '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', 'already', '', '', '', '', '', '', '', '', '', '', '',
                '', '', 'already', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                '', '', '', '', '', '', '', '', '', '', '', 'Provides', '', '',
                '', '', '', 'completed', '', '', '', '', '', '', '', '', '', ''
            }
        else:
            default_stopwords = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'about',
                'into', 'through', 'during', 'before', 'after', 'above', 'below', 'under', 'over', 'between', 'among',
                'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
                'doing', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'ought', 'i', 'you',
                'he', 'she', 'it', 'we', 'they', 'them', 'their', 'theirs', 'our', 'ours', 'your', 'yours', 'my', 'mine',
                'this', 'that', 'these', 'those', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both',
                'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
            }
        
        self.stopwords.update(default_stopwords)
        
        # ProvidesstopwordsFilePath, Loadcustom stopwords
        if self.stopwords_path:
            try:
                with open(self.stopwords_path, 'r', encoding='utf-8') as f:
                    custom_stopwords = {line.strip() for line in f if line.strip()}
                    self.stopwords.update(custom_stopwords)
                    # InitializeoutputInfo
                    if not TextProcessor._stopwords_initialized:
                        print(f"Successfully loaded custom stopwords, count: {len(custom_stopwords)}")
            except Exception as e:
                # outputErrorInfo
                print(f"Failed to load stopwords file: {e}")
        
        # InitializeoutputInfo
        if not TextProcessor._stopwords_initialized:
            print(f"stopwords initialization completed, total: {len(self.stopwords)}")
            # markalreadyInitialize
            TextProcessor._stopwords_initialized = True
    
    def clean_text(self, text: str, **kwargs) -> str:
        """
        Text
        
        Args:
            text: inputText
            remove_punctuation: removepunctuation
            remove_whitespace: removeextraEmpty
            lowercase: convert tolowercase
            remove_stopwords: removestopwords
            remove_numbers: removenumbers
            remove_urls: removeURL
            **kwargs: Parameter
            
        Returns:
            Text
        """
        if not isinstance(text, str):
            return ""
        
        # removeextraEmptycharacter
        if kwargs.get("remove_whitespace", True):
            text = re.sub(r'\s+', ' ', text)
        
        # removeURL
        if kwargs.get("remove_urls", True):
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        if kwargs.get("lowercase", True) and self.language == "en":
            text = text.lower()
        
        if kwargs.get("remove_numbers", False):
            text = re.sub(r'\d+', '', text)
        
        if kwargs.get("remove_punctuation", True):
            if self.language == "zh":
                text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)
            else:
                text = re.sub(r'[{}]'.format(re.escape(string.punctuation)), '', text)
        
        if kwargs.get("remove_stopwords", False):
            words = self.tokenize(text)
            words = [word for word in words if word not in self.stopwords]
            text = ' '.join(words)
        
        return text.strip()
    
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """
        Texttokenize
        
        Args:
            text: inputText
            use_jieba: jiebatokenize()
            **kwargs: Parameter
            
        Returns:
            wordlist
        """
        if not isinstance(text, str):
            return []
        
        if self.language == "zh" and kwargs.get("use_jieba", True):
            try:
                words = list(jieba.cut(text))
            except Exception as e:
                print(f"jieba tokenization failed: {e}")
                words = re.findall(r'[\u4e00-\u9fa5]+|[a-zA-Z0-9]+', text)
        else:
            words = re.findall(r'[a-zA-Z0-9]+', text)
        
        if kwargs.get("remove_stopwords", True):
            words = [word for word in words if word not in self.stopwords]
        
        min_length = kwargs.get("min_length", 1)
        words = [word for word in words if len(word) >= min_length]
        
        return words
    
    def extract_keywords(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """
        word
        
        Args:
            text: inputText
            top_k: returnwordcount
            method:  ('tfidf', 'textrank', 'count')
            **kwargs: Parameter
            
        Returns:
            wordlist, 'word''weight'
        """
        top_k = kwargs.get("top_k", 10)
        method = kwargs.get("method", "tfidf")
        
        if method == "textrank" and self.language == "zh":
            # jiebaTextRankword
            try:
                keywords = jieba.analyse.textrank(text, topK=top_k, withWeight=True)
                return [{'word': word, 'weight': weight} for word, weight in keywords]
            except Exception as e:
                print(f"TextRankword extraction failed: {e}")
                # TFIDF
                method = "tfidf"
        
        if method == "tfidf" and self.language == "zh":
            # jiebaTF-IDFword
            try:
                keywords = jieba.analyse.extract_tags(text, topK=top_k, withWeight=True)
                return [{'word': word, 'weight': weight} for word, weight in keywords]
            except Exception as e:
                print(f"TF-IDFword extraction failed: {e}")
                method = "count"
        
        words = self.tokenize(text, remove_stopwords=True)
        word_counts = Counter(words)
        total_words = sum(word_counts.values())
        
        keywords = []
        for word, count in word_counts.most_common(top_k):
            keywords.append({
                'word': word,
                'weight': count / total_words if total_words > 0 else 0
            })
        
        return keywords
    
    def split_text(self, text: str, **kwargs) -> List[str]:
        """
        Text
        
        Args:
            text: inputText
            max_length: Max
            split_by:  ('sentence', 'token', 'character')
            overlap: 
            **kwargs: Parameter
            
        Returns:
            Textlist
        """
        max_length = kwargs.get("max_length", 500)
        split_by = kwargs.get("split_by", "sentence")
        overlap = kwargs.get("overlap", 0)
        
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        
        if split_by == "sentence":
            if self.language == "zh":
                sentences = re.split(r'[.!？；]', text)
            else:
                sentences = re.split(r'[.!?;]', text)
            
            # , chunkmax_length
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                # chunkmax_length, Savechunk
                if len(current_chunk) + len(sentence) + 1 > max_length and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Process
                    if overlap > 0:
                        # overlapText
                        current_chunk = current_chunk[-overlap:] + " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
            
            # Savechunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
        
        elif split_by == "token":
            # token
            tokens = self.tokenize(text)
            current_chunk_tokens = []
            current_length = 0
            
            for token in tokens:
                token_length = len(token)
                if current_length + token_length + 1 > max_length and current_chunk_tokens:
                    chunks.append(' '.join(current_chunk_tokens))
                    # Process
                    if overlap > 0:
                        # token
                        overlap_tokens = []
                        overlap_length = 0
                        for t in reversed(current_chunk_tokens):
                            if overlap_length + len(t) + 1 <= overlap:
                                overlap_tokens.insert(0, t)
                                overlap_length += len(t) + 1
                            else:
                                break
                        current_chunk_tokens = overlap_tokens
                        current_length = overlap_length
                    else:
                        current_chunk_tokens = []
                        current_length = 0
                
                current_chunk_tokens.append(token)
                current_length += token_length + 1  # +1 Empty
            
            # Savechunk
            if current_chunk_tokens:
                chunks.append(' '.join(current_chunk_tokens))
        
        else:  # split_by == "character"
            i = 0
            while i < len(text):
                end = min(i + max_length, len(text))
                chunks.append(text[i:end])
                i = end - overlap
        
        return chunks
    
    def format_text(self, text: str, **kwargs) -> str:
        """
        Text
        
        Args:
            text: inputText
            line_length: Max
            justify: 
            **kwargs: Parameter
            
        Returns:
            Text
        """
        line_length = kwargs.get("line_length", 80)
        justify = kwargs.get("justify", False)
        
        # removeextraEmpty
        text = re.sub(r'\s+', ' ', text)
        
        lines = []
        words = text.split()
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + (1 if current_line else 0) > line_length:
                if justify and current_line:
                    if len(current_line.split()) > 1:
                        current_line = self._justify_line(current_line, line_length)
                lines.append(current_line)
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return "\n".join(lines)
    
    def _justify_line(self, line: str, line_length: int) -> str:
        """
        Text
        
        Args:
            line: Text
            line_length: 
            
        Returns:
            Text
        """
        words = line.split()
        if len(words) <= 1:
            return line
        
        total_spaces = line_length - sum(len(word) for word in words)
        gaps = len(words) - 1
        spaces_between_words = total_spaces // gaps
        extra_spaces = total_spaces % gaps
        
        result = ""
        for i, word in enumerate(words):
            result += word
            if i < gaps:
                # Empty
                spaces = spaces_between_words + (1 if i < extra_spaces else 0)
                result += " " * spaces
        
        return result
    
    def count_words(self, text: str, **kwargs) -> int:
        """
        word
        
        Args:
            text: inputText
            use_tokenize: tokenize()
            **kwargs: Parameter
            
        Returns:
            word
        """
        if kwargs.get("use_tokenize", True):
            return len(self.tokenize(text, **kwargs))
        else:
            # Empty
            return len(text.split())
    
    def count_sentences(self, text: str, **kwargs) -> int:
        """
        
        
        Args:
            text: inputText
            **kwargs: Parameter
            
        Returns:
            
        """
        if self.language == "zh":
            sentences = re.split(r'[.!？；]', text)
        else:
            sentences = re.split(r'[.!?;]', text)
        
        # Empty
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)
    
    def normalize_text(self, text: str, **kwargs) -> str:
        """
        Text
        character、Empty
        
        Args:
            text: inputText
            **kwargs: Parameter
            
        Returns:
            Text
        """
        full_to_half = {
            '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
            '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
            'Ａ': 'A', 'Ｂ': 'B', 'Ｃ': 'C', 'Ｄ': 'D', 'Ｅ': 'E',
            'Ｆ': 'F', 'Ｇ': 'G', 'Ｈ': 'H', 'Ｉ': 'I', 'Ｊ': 'J',
            'Ｋ': 'K', 'Ｌ': 'L', 'Ｍ': 'M', 'Ｎ': 'N', 'Ｏ': 'O',
            'Ｐ': 'P', 'Ｑ': 'Q', 'Ｒ': 'R', 'Ｓ': 'S', 'Ｔ': 'T',
            'Ｕ': 'U', 'Ｖ': 'V', 'Ｗ': 'W', 'Ｘ': 'X', 'Ｙ': 'Y',
            'Ｚ': 'Z', 'ａ': 'a', 'ｂ': 'b', 'ｃ': 'c', 'ｄ': 'd',
            'ｅ': 'e', 'ｆ': 'f', 'ｇ': 'g', 'ｈ': 'h', 'ｉ': 'i',
            'ｊ': 'j', 'ｋ': 'k', 'ｌ': 'l', 'ｍ': 'm', 'ｎ': 'n',
            'ｏ': 'o', 'ｐ': 'p', 'ｑ': 'q', 'ｒ': 'r', 'ｓ': 's',
            'ｔ': 't', 'ｕ': 'u', 'ｖ': 'v', 'ｗ': 'w', 'ｘ': 'x',
            'ｙ': 'y', 'ｚ': 'z', ', ': ',', '.': '.', '!': '!',
            '？': '?', '；': ';', ':': ':', '(': '(', ')': ')',
            '[': '[', ']': ']', '《': '<', '》': '>', '“': '"',
            '”': '"', '‘': "'", '’': "'", '、': ',', '～': '~',
            '－': '-', '—': '-', '…': '...', '　': ' '  # Empty
        }
        
        for full, half in full_to_half.items():
            text = text.replace(full, half)
        
        # Emptycharacter
        text = re.sub(r'\s+', ' ', text)
        
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
        
        return text.strip()
    
    def remove_html_tags(self, text: str) -> str:
        """
        removeHTML
        
        Args:
            text: inputText
            
        Returns:
            removeText
        """
        # HTMLremove
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)
    
    def get_text_features(self, text: str) -> Dict[str, Any]:
        """
        Text
        
        Args:
            text: inputText
            
        Returns:
            Text
        """
        features = {
            "total_characters": len(text),
            "total_words": self.count_words(text),
            "total_sentences": self.count_sentences(text),
            "avg_word_length": 0,
            "avg_sentence_length": 0,
            "keyword_density": {},
            "top_keywords": []
        }
        
        if features["total_words"] > 0:
            words = self.tokenize(text, remove_stopwords=False)
            features["avg_word_length"] = sum(len(word) for word in words) / features["total_words"]
        
        if features["total_sentences"] > 0:
            features["avg_sentence_length"] = features["total_words"] / features["total_sentences"]
        
        keywords = self.extract_keywords(text, top_k=10)
        features["top_keywords"] = keywords
        
        total_words = features["total_words"]
        if total_words > 0:
            for keyword in keywords:
                features["keyword_density"][keyword["word"]] = keyword["weight"] * total_words
        
        return features

from typing import List, Dict, Any
from collections import Counter
import re

class OCROptimizedCompressor:
    """
    OCR, 
    """
    
    def __init__(self):
        """
        InitializeOCROptimizedCompressor
        """
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"Initialize rerankerModel failed: {e}")
            self.reranker = None
        
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
        except Exception as e:
            print(f"Initialize spacyModel failed: {e}")
            self.nlp = None
    
    def compress(self, query: str, ocr_docs: List[Dict], target_tokens: int = 2000):
        """
        
        
        Args:
            query: character
            ocr_docs: OCRlist, "text"
            target_tokens: tokencount
            
        Returns:
            
        """
        # Level 1: OCRText()
        cleaned_docs = self._clean_ocr_noise(ocr_docs)  # 、
        
        # Level 2: (!)
        semantic_chunks = self._rebuild_semantic_chunks(cleaned_docs)  # token
        
        # Level 3: 
        reranked_chunks = self._rerank_and_dedup(query, semantic_chunks)  # +
        
        # Level 4: Info
        final_context = self._extract_high_density_info(
            query, reranked_chunks, target_tokens
        )
        
        return final_context
    
    def _clean_ocr_noise(self, docs: List[Dict]) -> List[Dict]:
        """
        OCR
        
        Args:
            docs: OCRlist
            
        Returns:
            list
        """
        cleaned = []
        for doc in docs:
            text = doc["text"]
            
            # 1. removeOCR(character)
            text = re.sub(r'[│━─—§¶×®©]{5,}', '', text)
            
            lines = text.split('\n')
            line_counter = Counter(lines)
            filtered_lines = [
                line for line in lines
                if line_counter[line] < 3 and len(line.strip()) > 10
            ]
            
            filtered_lines = [line for line in filtered_lines if len(line.split()) > 2]
            
            cleaned_text = '\n'.join(filtered_lines)
            
            # 4. Info
            original_tokens = len(text.split())
            cleaned_tokens = len(cleaned_text.split())
            
            if cleaned_tokens / original_tokens > 0.3:  # 30%
                cleaned.append({**doc, "text": cleaned_text, "cleaned": True})
        
        print(f"[OCR Clean] Reduced from {sum(len(d['text'].split()) for d in docs)} to {sum(len(d['text'].split()) for d in cleaned)} tokens")
        return cleaned
    
    def _rebuild_semantic_chunks(self, docs: List[Dict], chunk_size: int = 300) -> List[Dict]:
        """
        :tokenOCRText
        
        Args:
            docs: list
            chunk_size: 
            
        Returns:
            list
        """
        all_chunks = []
        chunk_id = 0
        
        for doc in docs:
            text = doc["text"]
            
            # OCRText"1.", "A.", "、"
            paragraph_markers = r'\n\s*(\d+\.|\(?[A-Z]\)|[]、)'
            sections = re.split(paragraph_markers, text)
            
            if len(sections) < 3:  # , 
                if self.nlp:
                    sentences = list(self.nlp(text).sents)
                    window_size = 5  # 5
                    
                    for i in range(0, len(sentences), window_size):
                        chunk_sents = sentences[i:i+window_size]
                        chunk_text = " ".join([s.text for s in chunk_sents])
                        
                        # chunk
                        if self._is_semantically_coherent(chunk_text):
                            all_chunks.append({
                                "text": chunk_text,
                                "doc_id": doc.get("id"),
                                "chunk_id": chunk_id,
                                "source": "sliding_window"
                            })
                            chunk_id += 1
                else:
                    # nlpModel, character
                    for i in range(0, len(text), chunk_size):
                        all_chunks.append({
                            "text": text[i:i+chunk_size],
                            "doc_id": doc.get("id"),
                            "chunk_id": chunk_id,
                            "source": "character_split"
                        })
                        chunk_id += 1
            else:
                for section in sections:
                    if len(section.split()) > 50:  # 
                        all_chunks.append({
                            "text": section.strip(),
                            "doc_id": doc.get("id"),
                            "chunk_id": chunk_id,
                            "source": "paragraph_marker"
                        })
                        chunk_id += 1
        
        print(f"[Chunking] Created {len(all_chunks)} semantic chunks")
        return all_chunks
    
    def _is_semantically_coherent(self, text: str) -> bool:
        """
        :Validword
        
        Args:
            text: inputText
            
        Returns:
            
        """
        if not self.nlp:
            return len(text.split()) > 10  # nlpModel, 
        
        doc = self.nlp(text)
        
        has_entities = len(doc.ents) > 0
        has_nouns = sum(1 for token in doc if token.pos_ in ["NOUN", "PROPN"]) >= 3
        
        valid_tokens = sum(1 for token in doc if token.is_alpha and not token.is_stop)
        total_tokens = sum(1 for token in doc if not token.is_space)
        
        return has_entities and has_nouns and (valid_tokens / max(total_tokens, 1) > 0.5)
    
    def _rerank_and_dedup(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        
        
        Args:
            query: character
            chunks: list
            
        Returns:
            list
        """
        if self.reranker:
            pairs = [(query, chunk["text"]) for chunk in chunks]
            scores = self.reranker.predict(pairs)
            
            scored_chunks = [
                {**chunk, "rerank_score": float(score)}
                for chunk, score in zip(chunks, scores)
            ]
        else:
            # rerankerModel, 
            scored_chunks = [
                {**chunk, "rerank_score": 1.0}
                for chunk in chunks
            ]
        
        unique_chunks = []
        seen_texts = []
        
        for chunk in sorted(scored_chunks, key=lambda x: x["rerank_score"], reverse=True):
            if not any(self._text_similarity(chunk["text"], seen) > 0.9 for seen in seen_texts):
                unique_chunks.append(chunk)
                seen_texts.append(chunk["text"])
        
        print(f"[Dedup] Removed {len(scored_chunks) - len(unique_chunks)} duplicate chunks")
        return unique_chunks[:10]  # top-10
    
    def _text_similarity(self, t1: str, t2: str) -> float:
        """
        Jaccard
        
        Args:
            t1: Text1
            t2: Text2
            
        Returns:
            Jaccard
        """
        set1 = set(t1.lower().split())
        set2 = set(t2.lower().split())
        if not set1 and not set2:
            return 0.0
        return len(set1.intersection(set2)) / len(set1.union(set2))
    
    def _extract_high_density_info(
        self, query: str, chunks: List[Dict], target_tokens: int
    ) -> List[Dict]:
        """
        Info
        
        Args:
            query: character
            chunks: list
            target_tokens: tokencount
            
        Returns:
            
        """
        # chunkstoken
        final_chunks = []
        total_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = len(chunk["text"].split())
            
            if total_tokens + chunk_tokens > target_tokens:
                # chunk
                remaining = target_tokens - total_tokens
                if remaining > 100:  # 100token
                    truncated = " ".join(chunk["text"].split()[:remaining])
                    final_chunks.append({**chunk, "text": truncated, "truncated": True})
                break
            
            final_chunks.append(chunk)
            total_tokens += chunk_tokens
        
        print(f"[Final] Compressed to {total_tokens} tokens from {len(chunks)} chunks")
        return final_chunks
    
    def compress_for_precision(self, query: str, ocr_text: str, max_tokens: int = 2000):
        """
        :numbers、word、list
        
        Args:
            query: character
            ocr_text: OCRText
            max_tokens: Maxtokencount
            
        Returns:
            Text
        """
        number_patterns = r'\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*(?:%|million|billion|kg|mm|°C|kW)?'
        numbers = re.findall(number_patterns, ocr_text)
        
        query_entities = self.extract_entities(query)
        
        table_markers = r'(Table|Figure||)\s*\d+.*?\n'
        table_contexts = re.findall(table_markers + r'.{0,500}', ocr_text, re.DOTALL)
        
        page_markers = r'Page\s+\d+.*?\n'
        pages = re.findall(page_markers + r'.{0,300}', ocr_text)
        
        critical_parts = numbers + table_contexts + pages
        
        for entity in query_entities:
            entity_pattern = re.compile(re.escape(entity), re.IGNORECASE)
            matches = entity_pattern.finditer(ocr_text)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(ocr_text), match.end() + 100)
                context = ocr_text[start:end]
                critical_parts.append(context)
        
        critical_text = "\n".join(critical_parts)
        
        # 6. token, Data
        if len(critical_text.split()) > max_tokens:
            critical_text = self.remove_descriptive_words(critical_text)
        
        return critical_text
    
    def extract_entities(self, query: str) -> List[str]:
        """
        
        
        Args:
            query: character
            
        Returns:
            list
        """
        entities = []
        
        if self.nlp:
            try:
                doc = self.nlp(query)
                entities = [ent.text for ent in doc.ents] + [
                    chunk.text for chunk in doc.noun_chunks
                    if any(token.pos_ in ["NOUN", "PROPN"] for token in chunk)
                ]
            except Exception as e:
                print(f"Failed: {e}")
                # nlpModel, word
                entities = re.findall(r'[A-Z][a-z]+|[A-Z]+[A-Z]?|[\u4e00-\u9fa5]+', query)
        else:
            # nlpModel, word
            entities = re.findall(r'[A-Z][a-z]+|[A-Z]+[A-Z]?|[\u4e00-\u9fa5]+', query)
        
        # Emptycharacter
        entities = list(set([ent.strip() for ent in entities if ent.strip()]))
        return entities
    
    def remove_descriptive_words(self, text: str) -> str:
        """
        , DataInfo
        
        Args:
            text: inputText
            
        Returns:
            ProcessText
        """
        if self.nlp:
            try:
                doc = self.nlp(text)
                keep_pos = ["NOUN", "PROPN", "VERB", "NUM", "SYM", "X"]
                filtered_tokens = [token.text for token in doc if token.pos_ in keep_pos or token.is_punct]
                return " ".join(filtered_tokens)
            except Exception as e:
                print(f"Failed to remove word: {e}")
                # nlpModel, 
                return re.sub(r'\b(?:very|extremely|highly|really|quite|somewhat|slightly|greatly|strongly|weakly|clearly|obviously|apparently|likely|probably|possibly)\b', '', text)
        else:
            # nlpModel, 
            return re.sub(r'\b(?:very|extremely|highly|really|quite|somewhat|slightly|greatly|strongly|weakly|clearly|obviously|apparently|likely|probably|possibly)\b', '', text)

# TextProcessorClassOCROptimizedCompressorClass
__all__ = ["TextProcessor", "OCROptimizedCompressor"]