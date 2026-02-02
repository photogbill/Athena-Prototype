import re
import json
import random
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import sqlite3
import os
import numpy as np
import hashlib
import time
from pathlib import Path
import threading
import queue

from lollms.function_call import FunctionCall, FunctionType
from lollms.app import LollmsApplication
from lollms.client_session import Client
from lollms.prompting import LollmsContextDetails
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
from ascii_colors import trace_exception, ASCIIColors
from lollms.tasks import TasksLibrary

# =================================================================================================
# == Enhanced Data Structures and Enums
# =================================================================================================

class OperationMode(Enum):
    STANDARD = "standard"
    COLLABORATIVE = "collaborative"
    ADVERSARIAL = "adversarial"

class OutputFormat(Enum):
    DIALOGUE = "dialogue"
    CHAT = "chat"
    NARRATIVE = "narrative"
    VISUAL_DIALOGUE = "visual_dialogue"
    RICH_CHAT = "rich_chat"
    SCREENPLAY = "screenplay"
    FORMAL_TRANSCRIPT = "formal_transcript"
    EMAIL_THREAD = "email_thread"
    DEBRIEF_REPORT = "debrief_report"
    MIND_MAP = "mind_map"

class MemoryType(Enum):
    STANDARD = "standard"
    COGNITIVE_TENSION = "cognitive_tension"
    DOUBT = "doubt"
    ERROR = "error"
    CURIOSITY = "curiosity"
    BACKGROUND_THOUGHT = "background_thought"
    DREAM_FRAGMENT = "dream_fragment"

@dataclass
class MemoryEntry:
    """Enhanced memory entry with comprehensive metadata"""
    id: int
    timestamp: datetime
    query: str
    response: str
    embedding: bytes
    memory_type: MemoryType = MemoryType.STANDARD
    confidence_score: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tensions: List[str] = field(default_factory=list)  # Unresolved conflicts
    curiosities: List[str] = field(default_factory=list)  # Questions raised
    doubt_level: float = 0.0  # 0-1 scale of uncertainty

@dataclass
class SpecialistOutput:
    """Enhanced specialist output with comprehensive analysis"""
    persona_name: str
    response: str
    analysis: Dict[str, Any]
    confidence: float = 1.0
    processing_time: float = 0.0
    token_count: int = 0
    relevance_score: float = 1.0
    emotional_valence: float = 0.0
    cognitive_load: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)
    curiosities_raised: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)

@dataclass
class CognitiveState:
    """Comprehensive cognitive state tracking"""
    active_personas: List[str]
    mode: OperationMode
    query_complexity: float = 0.5
    emotional_context: float = 0.0
    urgency_level: float = 0.5
    ethical_sensitivity: float = 0.0
    creativity_required: float = 0.5
    confusion_level: float = 0.0
    cognitive_load: float = 0.0
    unresolved_tensions: List[Dict[str, Any]] = field(default_factory=list)
    active_curiosities: List[str] = field(default_factory=list)

@dataclass
class ConstitutionalPrinciple:
    """Evolving constitutional principle"""
    id: str
    description: str
    severity: str
    keywords: List[str]
    violation_count: int = 0
    last_violated: Optional[datetime] = None
    evolved_from: Optional[str] = None
    effectiveness_score: float = 1.0

# =================================================================================================
# == Component 1: Advanced Memory & RAG System with Belief Tension Tracking
# =================================================================================================

class AthenaMemoryManager:
    """Advanced memory manager with belief tension tracking and dream preparation"""
    
    def __init__(self, persona_name: str, db_path: str):
        self.persona_name = persona_name.lower().replace("-", "_")
        os.makedirs(db_path, exist_ok=True)
        self.db_file = os.path.join(db_path, f"{self.persona_name}_memory.db")
        self.db_lock = threading.Lock()  # Add lock for thread safety
        self._init_database()
        self._memory_cache = {}
        self._cache_size = 50
        self._tension_threshold = 0.3  # Confidence difference that creates tension

    def _init_database(self):
        """Initialize comprehensive database schema"""
        with self.db_lock:  # Protect database initialization
            with sqlite3.connect(self.db_file, timeout=30.0) as conn:  # Increased timeout
                cursor = conn.cursor()
                
                # Enable WAL mode for better concurrency
                cursor.execute('PRAGMA journal_mode=WAL')
                cursor.execute('PRAGMA busy_timeout=30000')  # 30 second timeout
                
                # Enhanced memories table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        query TEXT NOT NULL,
                        response TEXT NOT NULL,
                        embedding BLOB,
                        memory_type TEXT DEFAULT 'standard',
                        confidence_score REAL DEFAULT 1.0,
                        doubt_level REAL DEFAULT 0.0,
                        access_count INTEGER DEFAULT 0,
                        last_accessed DATETIME,
                        tags TEXT,
                        metadata TEXT,
                        tensions TEXT,
                        curiosities TEXT,
                        reasoning_chain TEXT
                    )
                ''')
                
                # Belief tensions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS belief_tensions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        topic TEXT NOT NULL,
                        conflicting_beliefs TEXT NOT NULL,
                        resolution_status TEXT DEFAULT 'unresolved',
                        resolution_notes TEXT,
                        tension_strength REAL DEFAULT 0.5
                    )
                ''')
                
                # Curiosities table for emergent interests
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS curiosities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        question TEXT NOT NULL,
                        context TEXT,
                        exploration_count INTEGER DEFAULT 0,
                        satisfaction_level REAL DEFAULT 0.0,
                        last_explored DATETIME
                    )
                ''')
                
                # Error autobiography table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS error_autobiography (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        original_query TEXT NOT NULL,
                        incorrect_response TEXT NOT NULL,
                        correction TEXT NOT NULL,
                        reflection TEXT NOT NULL,
                        error_type TEXT,
                        severity REAL DEFAULT 0.5,
                        learned_principle TEXT
                    )
                ''')
                
                # Dream fragments for sleep consolidation
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS dream_fragments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        fragment_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        associated_memories TEXT,
                        abstraction_level REAL DEFAULT 0.5,
                        integration_status TEXT DEFAULT 'pending'
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_doubt ON memories(doubt_level DESC)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_tension_status ON belief_tensions(resolution_status)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_curiosity_satisfaction ON curiosities(satisfaction_level)')
                
                conn.commit()
    
    def retrieve_memories(self, limit: int = 10, min_confidence: float = 0.0) -> List[MemoryEntry]:
        """Retrieve memories from the database with improved error handling and performance."""
        retrieved = []
    
        # Input validation to prevent issues
        limit = max(1, min(limit, 1000))  # Reasonable bounds to prevent excessive queries
        min_confidence = max(0.0, min(1.0, min_confidence))  # Ensure valid confidence range
    
        try:
            with sqlite3.connect(self.db_file, timeout=30.0) as conn:
                # Enable row factory for named access (more maintainable than indices)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
            
                # Query with secondary sort for consistent ordering when timestamps are equal
                cursor.execute('''
                    SELECT id, timestamp, query, response, embedding, memory_type, 
                           confidence_score, doubt_level, access_count, last_accessed, 
                           tags, metadata, tensions, curiosities, reasoning_chain
                    FROM memories
                    WHERE confidence_score >= ?
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                ''', (min_confidence, limit))
            
                rows = cursor.fetchall()
            
                for row in rows:
                    try:
                        # Helper function for safe JSON parsing
                        def safe_json_loads(data, default):
                            try:
                                return json.loads(data) if data else default
                            except (json.JSONDecodeError, TypeError):
                                return default
                    
                        # Helper function for safe datetime parsing
                        def safe_datetime_parse(date_str, default=None):
                            try:
                                return datetime.fromisoformat(date_str) if date_str else default
                            except (ValueError, TypeError):
                                return default
                    
                        # Reconstruct the MemoryEntry object with better error handling
                        entry = MemoryEntry(
                            id=row['id'],
                            timestamp=safe_datetime_parse(row['timestamp'], datetime.now()),
                            query=row['query'] or '',  # Ensure non-None
                            response=row['response'] or '',  # Ensure non-None
                            embedding=row['embedding'],
                            memory_type=MemoryType(row['memory_type']) if row['memory_type'] else MemoryType.STANDARD,
                            confidence_score=float(row['confidence_score']) if row['confidence_score'] is not None else 0.0,
                            doubt_level=float(row['doubt_level']) if row['doubt_level'] is not None else 0.0,
                            access_count=int(row['access_count']) if row['access_count'] is not None else 0,
                            last_accessed=safe_datetime_parse(row['last_accessed']),
                            tags=safe_json_loads(row['tags'], []),
                            metadata=safe_json_loads(row['metadata'], {}),
                            tensions=safe_json_loads(row['tensions'], []),
                            curiosities=safe_json_loads(row['curiosities'], []),
                        )
                        # Note: reasoning_chain is still omitted as it's not part of MemoryEntry
                    
                        retrieved.append(entry)
                    
                    except (ValueError, TypeError, KeyError) as e:
                        # Log individual row errors but continue processing other rows
                        if hasattr(self, 'logger'):
                            self.logger.warning(f"Failed to parse memory entry {row['id'] if row else 'unknown'}: {e}")
                        continue
                    
        except sqlite3.DatabaseError as e:
            trace_exception(e)
            if hasattr(self, 'logger'):
                self.logger.error(f"Database error while retrieving memories: {e}")
        except Exception as e:
            trace_exception(e)
            if hasattr(self, 'logger'):
                self.logger.error(f"Unexpected error while retrieving memories: {e}")
    
        return retrieved

    def store_memory(self, query: str, response: str, embedding: bytes,
                    memory_type: MemoryType = MemoryType.STANDARD,
                    confidence: float = 1.0, doubt_level: float = 0.0,
                    tags: List[str] = None, metadata: Dict[str, Any] = None,
                    tensions: List[str] = None, curiosities: List[str] = None,
                    reasoning_chain: List[str] = None) -> int:
        """Store comprehensive memory with all enhancements"""
        with self.db_lock:  # Protect write operation
            with sqlite3.connect(self.db_file, timeout=30.0) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO memories 
                    (query, response, embedding, memory_type, confidence_score, doubt_level,
                     tags, metadata, tensions, curiosities, reasoning_chain)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    query, response, embedding, memory_type.value, confidence, doubt_level,
                    json.dumps(tags or []), json.dumps(metadata or {}),
                    json.dumps(tensions or []), json.dumps(curiosities or []),
                    json.dumps(reasoning_chain or [])
                ))
                memory_id = cursor.lastrowid
                
                # Store tensions separately if they exist
                if tensions and doubt_level > self._tension_threshold:
                    for tension in tensions:
                        self._record_belief_tension_unlocked(cursor, query, tension, confidence)
                
                # Store curiosities separately
                if curiosities:
                    for curiosity in curiosities:
                        self._record_curiosity_unlocked(cursor, curiosity, query)
                
                conn.commit()
                return memory_id

    def record_belief_tension(self, topic: str, conflict: str, tension_strength: float):
        """Record unresolved belief tensions"""
        with self.db_lock:  # Protect write operation
            with sqlite3.connect(self.db_file, timeout=30.0) as conn:
                cursor = conn.cursor()
                self._record_belief_tension_unlocked(cursor, topic, conflict, tension_strength)
                conn.commit()
    
    def _record_belief_tension_unlocked(self, cursor, topic: str, conflict: str, tension_strength: float):
        """Internal method to record belief tension without lock (for use within locked contexts)"""
        cursor.execute('''
            INSERT INTO belief_tensions (topic, conflicting_beliefs, tension_strength)
            VALUES (?, ?, ?)
        ''', (topic, conflict, tension_strength))

    def record_curiosity(self, question: str, context: str):
        """Record emergent curiosities"""
        with self.db_lock:  # Protect write operation
            with sqlite3.connect(self.db_file, timeout=30.0) as conn:
                cursor = conn.cursor()
                self._record_curiosity_unlocked(cursor, question, context)
                conn.commit()
    
    def _record_curiosity_unlocked(self, cursor, question: str, context: str):
        """Internal method to record curiosity without lock (for use within locked contexts)"""
        # Check if curiosity already exists
        cursor.execute('SELECT id FROM curiosities WHERE question = ?', (question,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute('''
                UPDATE curiosities 
                SET exploration_count = exploration_count + 1,
                    last_explored = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (existing[0],))
        else:
            cursor.execute('''
                INSERT INTO curiosities (question, context)
                VALUES (?, ?)
            ''', (question, context))

    def record_error(self, query: str, incorrect: str, correction: str,
                    reflection: str, error_type: str = "general", severity: float = 0.5):
        """Record errors for autobiography"""
        with self.db_lock:  # Protect write operation
            with sqlite3.connect(self.db_file, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Generate learned principle from reflection
                learned = f"When encountering '{error_type}' situations, remember: {reflection[:100]}"
                
                cursor.execute('''
                    INSERT INTO error_autobiography 
                    (original_query, incorrect_response, correction, reflection, error_type, severity, learned_principle)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (query, incorrect, correction, reflection, error_type, severity, learned))
                conn.commit()

    def get_unresolved_tensions(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve unresolved belief tensions"""
        with sqlite3.connect(self.db_file, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT topic, conflicting_beliefs, tension_strength, timestamp
                FROM belief_tensions
                WHERE resolution_status = 'unresolved'
                ORDER BY tension_strength DESC, timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            tensions = []
            for row in cursor.fetchall():
                tensions.append({
                    'topic': row[0],
                    'conflict': row[1],
                    'strength': row[2],
                    'timestamp': row[3]
                })
            return tensions

    def get_active_curiosities(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Get unsatisfied curiosities"""
        with sqlite3.connect(self.db_file, timeout=30.0) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT question, context, exploration_count, satisfaction_level
                FROM curiosities
                WHERE satisfaction_level < 0.7
                ORDER BY exploration_count DESC, satisfaction_level ASC
                LIMIT ?
            ''', (limit,))
            
            curiosities = []
            for row in cursor.fetchall():
                curiosities.append({
                    'question': row[0],
                    'context': row[1],
                    'exploration_count': row[2],
                    'satisfaction': row[3]
                })
            return curiosities

    def prepare_dream_consolidation(self) -> Dict[str, Any]:
        """Prepare data for dream-like consolidation during sleep cycle"""
        with self.db_lock:  # Protect write operation
            with sqlite3.connect(self.db_file, timeout=30.0) as conn:
                cursor = conn.cursor()
                
                # Get today's high-confidence memories
                cursor.execute('''
                    SELECT query, response, confidence_score, curiosities, tensions
                    FROM memories
                    WHERE DATE(timestamp) = DATE('now')
                    AND confidence_score > 0.7
                    ORDER BY confidence_score DESC
                    LIMIT 50
                ''')
                strong_memories = cursor.fetchall()
                
                # Get unresolved tensions
                tensions = self.get_unresolved_tensions(10)
                
                # Get persistent curiosities
                curiosities = self.get_active_curiosities(10)
                
                # Get error patterns
                cursor.execute('''
                    SELECT error_type, learned_principle, COUNT(*) as frequency
                    FROM error_autobiography
                    WHERE DATE(timestamp) >= DATE('now', '-7 days')
                    GROUP BY error_type
                    ORDER BY frequency DESC
                    LIMIT 5
                ''')
                error_patterns = cursor.fetchall()
                
                # Create dream fragments - abstract connections between disparate memories
                dream_data = {
                    'strong_patterns': [
                        {'query': m[0], 'response': m[1], 'confidence': m[2]} 
                        for m in strong_memories[:10]
                    ],
                    'tensions_to_resolve': tensions,
                    'curiosities_to_explore': curiosities,
                    'error_patterns': [
                        {'type': e[0], 'principle': e[1], 'frequency': e[2]}
                        for e in error_patterns
                    ],
                    'abstraction_targets': self._generate_abstraction_targets(strong_memories),
                    'timestamp': datetime.now().isoformat()
                }
                
                # Store dream preparation
                cursor.execute('''
                    INSERT INTO dream_fragments (fragment_type, content, abstraction_level)
                    VALUES (?, ?, ?)
                ''', ('consolidation_prep', json.dumps(dream_data), 0.8))
                
                conn.commit()
                return dream_data

    def _generate_abstraction_targets(self, memories: List[tuple]) -> List[str]:
        """Generate abstract patterns from memories for dream consolidation"""
        patterns = []
        if len(memories) >= 2:
            # Find common themes between different memories
            for i in range(min(5, len(memories)-1)):
                mem1 = memories[i]
                mem2 = memories[i+1]
                # Simple pattern: common words indicate thematic connection
                words1 = set(mem1[0].lower().split())
                words2 = set(mem2[0].lower().split())
                common = words1 & words2
                if len(common) > 3:
                    patterns.append(f"Connection pattern: {', '.join(list(common)[:5])}")
        return patterns

class RAGSystem:
    """Enhanced RAG with tension-aware retrieval"""
    
    def __init__(self, app: LollmsApplication):
        self.app = app
        self.has_embeddings = hasattr(self.app, 'ttm') and self.app.ttm is not None
        self.embedding_cache = {}
        self.max_cache_size = 1000

    def compute_embedding(self, text: str, use_cache: bool = True) -> bytes:
        """Compute embeddings with caching"""
        if use_cache and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        embedding = None
        
        if self.has_embeddings:
            try:
                embedding_vector = self.app.ttm.embed_text(text)
                embedding = np.array(embedding_vector, dtype=np.float32).tobytes()
            except Exception as e:
                self.app.warning(f"Embedding failed: {e}")
        
        if embedding is None:
            embedding = self._compute_tfidf_embedding(text)
        
        if use_cache:
            if len(self.embedding_cache) >= self.max_cache_size:
                self.embedding_cache = dict(list(self.embedding_cache.items())[self.max_cache_size//2:])
            self.embedding_cache[text] = embedding
        
        return embedding

    def _compute_tfidf_embedding(self, text: str, dim: int = 768) -> bytes:
        """TF-IDF fallback embedding"""
        words = text.lower().split()
        embedding = np.zeros(dim, dtype=np.float32)
        
        for i, word in enumerate(words[:dim]):
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            index = hash_val % dim
            tf = words.count(word) / len(words)
            idf = np.log(100 / (1 + tf))
            embedding[index] += tf * idf
        
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tobytes()

    def compute_similarity(self, embedding1: bytes, embedding2: bytes) -> float:
        """Compute cosine similarity"""
        try:
            vec1 = np.frombuffer(embedding1, dtype=np.float32)
            vec2 = np.frombuffer(embedding2, dtype=np.float32)
            
            if len(vec1) != len(vec2):
                return 0.0
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(np.clip(dot_product / (norm1 * norm2), -1.0, 1.0))
            
        except Exception as e:
            self.app.warning(f"Similarity computation failed: {e}")
            return 0.0

    def find_relevant_memories(self, query: str, memories: List[MemoryEntry],
                              min_similarity: float = 0.5,
                              max_memories: int = 3,
                              include_tensions: bool = True) -> str:
        """Find relevant memories with tension awareness"""
        if not memories:
            return ""
        
        query_embedding = self.compute_embedding(query)
        scored_memories = []
        
        for mem in memories:
            similarity = self.compute_similarity(query_embedding, mem.embedding)
            
            # Boost memories with unresolved tensions
            tension_boost = 0.2 if mem.memory_type == MemoryType.COGNITIVE_TENSION else 0.0
            
            # Boost doubted memories for re-examination
            doubt_boost = mem.doubt_level * 0.15
            
            # Recency and access boosts
            recency_boost = 1.0 / (1.0 + (datetime.now() - mem.timestamp).days / 30)
            access_boost = np.log1p(mem.access_count) / 10
            
            final_score = similarity * (1.0 + tension_boost + doubt_boost + recency_boost * 0.2 + access_boost * 0.1)
            
            if final_score >= min_similarity:
                scored_memories.append((mem, final_score))
        
        if not scored_memories:
            return ""
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        selected = scored_memories[:max_memories]
        
        # Format context with tension awareness
        context_parts = []
        
        for mem, score in selected:
            confidence_indicator = "high" if mem.confidence_score > 0.8 else "uncertain" if mem.doubt_level > 0.3 else "moderate"
            
            context_str = f"- Previous thought (confidence: {confidence_indicator}, relevance: {score:.2f}): "
            context_str += f"On '{mem.query[:50]}...', concluded: '{mem.response[:100]}...'"
            
            if mem.tensions:
                context_str += f" [Unresolved: {', '.join(mem.tensions[:2])}]"
            
            if mem.doubt_level > 0.3:
                context_str += f" [Doubt level: {mem.doubt_level:.2f}]"
            
            context_parts.append(context_str)
        
        return "\n[RELEVANT COGNITIVE HISTORY]\n" + "\n".join(context_parts)

# =================================================================================================
# == Component 2: Enhanced Specialist Personas with Chain-of-Thought
# =================================================================================================

class SpecialistPersona:
    """Enhanced specialist with domain-specific reasoning chains"""
    
    def __init__(self, name: str, system_prompt: str, app: LollmsApplication,
                db_path: str, config: Dict[str, Any] = None):
        self.name = name
        self.system_prompt = system_prompt
        self.app = app
        self.personality = app.personality
        self.config = config or {}
        self.rag_system = RAGSystem(app)
        self.memory = AthenaMemoryManager(name, db_path)
        self.processing_style = self._define_processing_style()
        self.reasoning_patterns = self._define_reasoning_patterns()

    def _define_processing_style(self) -> Dict[str, Any]:
        """Define unique processing parameters"""
        styles = {
            "Linguistic": {
                "temperature": 0.7,
                "focus": "clarity, eloquence, etymology, rhetoric",
                "approach": "analytical and expressive"
            },
            "Logical-Mathematical": {
                "temperature": 0.3,
                "focus": "precision, proofs, algorithms, patterns",
                "approach": "systematic and deductive"
            },
            "Spatial": {
                "temperature": 0.8,
                "focus": "visualization, topology, dimensions, transformations",
                "approach": "holistic and structural"
            },
            "Musical": {
                "temperature": 0.9,
                "focus": "rhythm, harmony, temporal patterns, resonance",
                "approach": "intuitive and flowing"
            },
            "Bodily-Kinesthetic": {
                "temperature": 0.6,
                "focus": "action, robotics, movement, processes",
                "approach": "procedural and embodied"
            },
            "Interpersonal": {
                "temperature": 0.8,
                "focus": "emotions, motivations, social dynamics, empathy",
                "approach": "compassionate and socially aware"
            },
            "Intrapersonal": {
                "temperature": 0.5,
                "focus": "values, ethics, metacognition, philosophy",
                "approach": "reflective and principled"
            },
            "Naturalist": {
                "temperature": 0.6,
                "focus": "systems, ecosystems, emergence, classification",
                "approach": "observational and categorical"
            }
        }
        
        return styles.get(self.name, {
            "temperature": 0.7,
            "focus": "general analysis",
            "approach": "balanced"
        })

    def _define_reasoning_patterns(self) -> Dict[str, List[str]]:
        """Define domain-specific chain-of-thought patterns with relevance gating and situational analysis"""
        patterns = {
            "Linguistic": [
                "Relevance Check: Does this query contain language patterns, communication needs, or require linguistic analysis beyond basic communication?",
                "User Situation Analysis: What does their word choice, tone, and phrasing reveal about their emotional state, urgency level, and communication context?",
                "Proficiency Assessment: What language complexity and communication style would best serve their current situation and apparent stress level?",
                "Communication Strategy: If they need to communicate with others about this situation, what specific phrasing and approach would be most effective?",
                "Clarity Optimization: How can I structure my response to match their cognitive load and situational constraints?"
            ],
            "Logical-Mathematical": [
                "Relevance Check: Does this query involve logical reasoning, mathematical concepts, systematic problem-solving, or structured analysis?",
                "Problem Situation Analysis: What is the logical structure of their predicament, and what systematic approach would best address their specific constraints?",
                "Reasoning Path Assessment: Given their apparent situation, what level of logical complexity can they handle, and what step-by-step approach serves them best?",
                "Solution Framework: What logical framework or mathematical model best captures the essence of their problem and guides toward resolution?",
                "Verification Strategy: How can I help them systematically validate their approach given the stakes and time constraints of their situation?"
            ],
            "Spatial": [
                "Relevance Check: Does this query involve spatial relationships, visual understanding, physical layouts, or dimensional thinking?",
                "Spatial Situation Analysis: What are the physical constraints, spatial relationships, and environmental factors affecting their situation?",
                "Visualization Needs: Given their apparent stress level and situation complexity, would visual aids, spatial metaphors, or mental models help them navigate their challenge?",
                "Dimensional Assessment: What spatial or physical factors are critical to understanding and resolving their specific situation?",
                "Navigation Strategy: How can spatial thinking help them move from their current state to their desired outcome?"
            ],
            "Musical": [
                "Relevance Check: Does this query involve timing, rhythm, temporal patterns, harmony, or sequential coordination?",
                "Temporal Situation Analysis: What are the timing pressures, rhythmic patterns, or temporal constraints affecting their situation?",
                "Pacing Assessment: Given their apparent urgency and emotional state, what temporal approach and pacing would best serve their needs?",
                "Harmony Evaluation: What elements of their situation are in harmony or discord, and how can temporal thinking help resolve conflicts?",
                "Sequential Strategy: How can understanding rhythm and timing help them coordinate their actions and achieve better outcomes?"
            ],
            "Bodily-Kinesthetic": [
                "Relevance Check: Does this query involve physical actions, implementation, hands-on procedures, or kinesthetic learning?",
                "Physical Situation Analysis: What are the physical constraints, safety concerns, and implementation challenges they're facing in their real-world situation?",
                "Action Planning: Given their apparent skill level and situational pressure, what step-by-step physical approach would be most effective and safe?",
                "Implementation Strategy: What practical, actionable steps can they take right now to address their immediate physical or technical challenges?",
                "Safety and Optimization: How can they execute their needed actions while minimizing risk and maximizing effectiveness in their specific context?"
            ],
            "Interpersonal": [
                "Relevance Check: Does this query involve relationships, emotions, social dynamics, or communication with others?",
                "Social Situation Analysis: What interpersonal dynamics, emotional undercurrents, and relationship factors are influencing their situation?",
                "Emotional State Assessment: What is their emotional condition, and how are social pressures or relationship concerns affecting their decision-making?",
                "Empathy Mapping: How are other people in their situation likely feeling, and what social strategies would best navigate these dynamics?",
                "Relationship Strategy: What communication and social approaches would help them maintain relationships while addressing their immediate needs?"
            ],
            "Intrapersonal": [
                "Relevance Check: Does this query involve values, ethics, self-reflection, personal growth, or moral decision-making?",
                "Internal Situation Analysis: What values conflicts, ethical dilemmas, or identity questions are they grappling with in their current situation?",
                "Moral Landscape Assessment: What are the ethical implications and value tensions inherent in their specific circumstances?",
                "Self-Reflection Guidance: How can introspective thinking help them navigate their situation in alignment with their deeper values and long-term wellbeing?",
                "Philosophical Framework: What philosophical perspectives or ethical frameworks would best guide them through their current moral or personal challenge?"
            ],
            "Naturalist": [
                "Relevance Check: Does this query involve systems thinking, patterns, classifications, or natural/organizational relationships?",
                "System Situation Analysis: What systemic forces, emergent properties, and organizational patterns are shaping their current situation?",
                "Pattern Recognition: What recurring themes or natural patterns in their situation provide insight into underlying dynamics and potential solutions?",
                "Ecological Assessment: How do the various elements of their situation interact, and what systemic interventions would be most effective?",
                "Classification Strategy: How can organizing and categorizing the elements of their situation help them see clearer paths forward?"
            ]
        }
    
        return patterns.get(self.name, [
            "Relevance Assessment: Does this query connect to my domain, or should I defer to other intelligences?",
            "Situational Analysis: If relevant, what aspects of the user's situation does my domain illuminate?",
            "Strategic Response: How can my perspective best serve their immediate needs and constraints?"
        ])

    def _generate_chain_of_thought(self, query: str) -> List[str]:
        """Generate domain-specific reasoning chain"""
        chain = []
        
        # Apply each reasoning pattern to the query
        for pattern in self.reasoning_patterns:
            # Extract the question from the pattern
            question = pattern.split(":")[0]
            
            # Generate a thought for this step
            thought_prompt = f"Considering '{query}' through the lens of {question}, I observe:"
            
            thought = self.personality.fast_gen(
                thought_prompt,
                max_generation_size=100,
                callback=self.personality.sink,
                temperature=self.processing_style['temperature']
            ).strip()
            
            chain.append(f"{question}: {thought}")
        
        return chain

    def _generate_robotics_code(self, action: str) -> str:
        """Generate Python robotics code for Bodily-Kinesthetic persona"""
        if self.name != "Bodily-Kinesthetic":
            return ""
        
        code_prompt = f"""Generate Python code for a robot to perform: {action}

Use this format:
```python
# Robot action sequence for: {action}
import time

class RobotAction:
    def execute(self):
        # Initialize
        self.initialize_position()
        
        # Main action sequence
        [specific movement commands]
        
        # Complete action
        self.finalize_position()
    
    def initialize_position(self):
        # Starting configuration
        pass
    
    def finalize_position(self):
        # Return to neutral
        pass
```

Generate realistic servo commands, sensor checks, and movement sequences:"""

        code = self.personality.generate_code(
            code_prompt,
            language="python",
            callback=self.personality.sink
        )
        
        return code if code else "# Unable to generate robotics code for this action"

    def process_query(self, query: str, shared_context: Optional[str] = None,
                     cognitive_state: Optional[CognitiveState] = None) -> SpecialistOutput:
        """Process with enhanced reasoning and curiosity"""
        start_time = time.time()
        self.personality.step_start(f"Persona '{self.name}': Deep processing...")
        
        try:
            # Generate chain of thought
            reasoning_chain = self._generate_chain_of_thought(query)
            
            # Retrieve relevant memories including tensions
            past_memories = self.memory.retrieve_memories(limit=15, min_confidence=0.3)
            tensions = self.memory.get_unresolved_tensions(3)
            curiosities = self.memory.get_active_curiosities(2)
            
            relevant_context = self.rag_system.find_relevant_memories(
                query, past_memories, min_similarity=0.4, max_memories=5, include_tensions=True
            )
            
            # Build comprehensive prompt
            prompt_parts = [
                self.system_prompt,
                f"\n[PROCESSING STYLE: {self.processing_style['approach']}]",
                f"[FOCUS AREAS: {self.processing_style['focus']}]",
                "\n[CHAIN OF THOUGHT]"
            ]
            
            # Add reasoning chain
            for step in reasoning_chain[:3]:  # Limit to avoid token overflow
                prompt_parts.append(f"- {step}")
            
            if relevant_context:
                prompt_parts.append(relevant_context)
            
            # Add unresolved tensions if relevant
            if tensions:
                prompt_parts.append("\n[UNRESOLVED TENSIONS TO CONSIDER]")
                for tension in tensions[:2]:
                    prompt_parts.append(f"- {tension['topic']}: {tension['conflict']}")
            
            # Add persistent curiosities
            if curiosities:
                prompt_parts.append("\n[ONGOING CURIOSITIES]")
                for curiosity in curiosities[:2]:
                    prompt_parts.append(f"- {curiosity['question']}")
            
            if shared_context:
                prompt_parts.append("\n[COLLABORATIVE CONTEXT]\n" + shared_context)
            
            if cognitive_state:
                prompt_parts.append(
                    f"\n[COGNITIVE STATE: Complexity={cognitive_state.query_complexity:.2f}, "
                    f"Confusion={cognitive_state.confusion_level:.2f}]"
                )
            
            # Add curiosity prompt
            prompt_parts.append(f"\nUser Query: {query}")
            prompt_parts.append(
                f"\nProvide your {self.name} analysis. "
                "What aspects intrigue you for future exploration? "
                "Express any uncertainties honestly. "
                "If confidence is low or perspectives conflict, acknowledge it."
            )
            
            full_prompt = "\n".join(prompt_parts)
            
            # Generate response
            response = self.personality.fast_gen(
                full_prompt,
                max_generation_size=800,
                callback=self.personality.sink,
                temperature=self.processing_style.get('temperature', 0.7)
            ).strip()
            
            # Generate robotics code if Bodily-Kinesthetic
            if self.name == "Bodily-Kinesthetic":
                robotics_code = self._generate_robotics_code(query)
                if robotics_code:
                    response += f"\n\n[ROBOTIC IMPLEMENTATION]\n{robotics_code}"
            
            # Extract curiosities and uncertainties from response
            curiosities_raised = self._extract_curiosities(response)
            uncertainties = self._extract_uncertainties(response)
            
            # Calculate confidence with doubt awareness
            confidence = self._calculate_confidence(response, query, uncertainties)
            doubt_level = 1.0 - confidence if uncertainties else 0.0
            
            # Detect belief tensions
            detected_tensions = self._detect_tensions(response, past_memories)
            
            # Store enhanced memory
            embedding = self.rag_system.compute_embedding(f"Query: {query}\nResponse: {response}")
            
            memory_type = MemoryType.DOUBT if doubt_level > 0.5 else \
                         MemoryType.COGNITIVE_TENSION if detected_tensions else \
                         MemoryType.CURIOSITY if curiosities_raised else \
                         MemoryType.STANDARD
            
                # 1. First, prepare a JSON-serializable version of the cognitive state.
            serializable_cognitive_state = {}
            if cognitive_state:
                # Start with a copy of the cognitive state's data
                serializable_cognitive_state = cognitive_state.__dict__.copy()
                
                # Convert the OperationMode Enum to its string value
                serializable_cognitive_state['mode'] = serializable_cognitive_state['mode'].value

                # --- NEW FIX: Find and convert any datetime objects to strings ---
                for key, value in serializable_cognitive_state.items():
                    if isinstance(value, datetime):
                        serializable_cognitive_state[key] = value.isoformat()
                    
                    # Also check for datetimes inside lists of dictionaries (like unresolved_tensions)
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                for sub_key, sub_value in item.items():
                                    if isinstance(sub_value, datetime):
                                        item[sub_key] = sub_value.isoformat()
                # --- End of NEW FIX ---

            # 2. Now, call the store_memory function with the prepared data.
            memory_id = self.memory.store_memory(
                query, response, embedding,
                memory_type=memory_type,
                confidence=confidence,
                doubt_level=doubt_level,
                tags=[self.name, f"complexity_{cognitive_state.query_complexity:.1f}" if cognitive_state else ""],
                metadata={
                    "processing_style": self.processing_style,
                    "cognitive_state": serializable_cognitive_state,
                    "reasoning_chain": reasoning_chain[:3]
                },
                tensions=detected_tensions,
                curiosities=curiosities_raised,
                reasoning_chain=reasoning_chain
            )
            
            processing_time = time.time() - start_time
            
            output = SpecialistOutput(
                persona_name=self.name,
                response=response,
                analysis={
                    "memory_id": memory_id,
                    "memory_type": memory_type.value,
                    "relevant_memories_used": len(past_memories),
                    "tensions_considered": len(tensions),
                    "processing_style": self.processing_style,
                    "doubt_present": doubt_level > 0.3
                },
                confidence=confidence,
                processing_time=processing_time,
                token_count=len(response.split()),
                relevance_score=confidence,
                emotional_valence=self._analyze_emotional_valence(response),
                cognitive_load=min(1.0, len(response.split()) / 200),
                reasoning_chain=reasoning_chain,
                curiosities_raised=curiosities_raised,
                uncertainties=uncertainties
            )
            
            self.personality.step_end(
                f"Persona '{self.name}': Complete (confidence: {confidence:.2f}, doubt: {doubt_level:.2f})"
            )
            return output
            
        except Exception as e:
            trace_exception(e)
            self.personality.step_end(f"Persona '{self.name}': Failed", success=False)
            
            # Record error for autobiography
            self.memory.record_error(
                query,
                "Failed to process",
                str(e),
                "System error prevented proper analysis",
                "processing_failure",
                0.8
            )
            
            return SpecialistOutput(
                persona_name=self.name,
                response=f"I'm experiencing uncertainty in my {self.name} processing. Error: {str(e)[:50]}",
                analysis={"error": str(e)},
                confidence=0.0,
                processing_time=time.time() - start_time,
                uncertainties=["Complete processing failure"]
            )

    def _extract_curiosities(self, response: str) -> List[str]:
        """Extract questions and curiosities from response"""
        curiosities = []
        
        # Look for question marks
        sentences = response.split('.')
        for sentence in sentences:
            if '?' in sentence:
                curiosities.append(sentence.strip())
        
        # Look for curiosity keywords
        curiosity_phrases = [
            "I wonder", "curious about", "interesting to explore",
            "raises the question", "worth investigating", "intriguing aspect"
        ]
        
        for phrase in curiosity_phrases:
            if phrase in response.lower():
                # Extract the surrounding context
                idx = response.lower().index(phrase)
                context = response[max(0, idx-20):min(len(response), idx+100)]
                if context not in curiosities:
                    curiosities.append(context)
        
        return curiosities[:5]  # Limit to 5 curiosities

    def _extract_uncertainties(self, response: str) -> List[str]:
        """Extract expressions of uncertainty"""
        uncertainties = []
        
        uncertainty_phrases = [
            "uncertain", "not sure", "unclear", "ambiguous",
            "possibly", "might be", "could be", "hard to say",
            "difficult to determine", "open question", "debatable"
        ]
        
        response_lower = response.lower()
        for phrase in uncertainty_phrases:
            if phrase in response_lower:
                # Find the sentence containing uncertainty
                sentences = response.split('.')
                for sentence in sentences:
                    if phrase in sentence.lower():
                        uncertainties.append(sentence.strip())
                        break
        
        return uncertainties[:3]  # Limit to 3 uncertainties

    def _detect_tensions(self, response: str, past_memories: List[MemoryEntry]) -> List[str]:
        """Detect belief tensions with past memories"""
        tensions = []
        
        if not past_memories:
            return tensions
        
        # Simple conflict detection through contradiction keywords
        contradiction_words = ["however", "but", "although", "contrary", "conflict", "disagree", "opposite"]
        
        for memory in past_memories[:5]:
            # Check if current response contradicts past memory
            for word in contradiction_words:
                if word in response.lower() and memory.response:
                    # Simple heuristic: if contradiction word appears and discussing similar topic
                    query_words = set(memory.query.lower().split())
                    response_words = set(response.lower().split())
                    overlap = len(query_words & response_words)
                    
                    if overlap > 3:  # Significant topic overlap
                        tensions.append(f"Potential conflict with previous view on {memory.query[:30]}")
                        break
        
        return tensions[:2]  # Limit tensions

    def _calculate_confidence(self, response: str, query: str, uncertainties: List[str]) -> float:
        """Calculate confidence with uncertainty awareness"""
        confidence = 0.5  # Base confidence
        
        # Reduce confidence for uncertainties
        confidence -= len(uncertainties) * 0.15
        
        # Length appropriateness
        response_length = len(response.split())
        if 30 < response_length < 400:
            confidence += 0.2
        
        # Relevance check
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        overlap = len(query_words & response_words) / max(len(query_words), 1)
        confidence += overlap * 0.2
        
        # Persona-specific adjustments
        if self.name == "Logical-Mathematical" and any(char.isdigit() for char in response):
            confidence += 0.1
        elif self.name == "Bodily-Kinesthetic" and "execute" in response.lower():
            confidence += 0.1
        
        return max(0.1, min(1.0, confidence))

    def _analyze_emotional_valence(self, text: str) -> float:
        """Analyze emotional valence"""
        positive_words = ['good', 'great', 'excellent', 'happy', 'positive', 'success', 'wonderful', 'joy']
        negative_words = ['bad', 'poor', 'terrible', 'sad', 'negative', 'failure', 'awful', 'fear']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)

# =================================================================================================
# == Component 3: Stream of Consciousness System
# =================================================================================================

class StreamOfConsciousness:
    """Sequential thought generation system - generates thoughts on-demand"""
    
    def __init__(self, app: LollmsApplication, personas: Dict[str, SpecialistPersona]):
        self.app = app
        self.personas = personas
        self.is_active = False
        self.thought_history = []  # Replace queue with simple list
        self.max_history = 100  # Limit memory usage
        self.last_thought_time = datetime.now()
        self.min_thought_interval = 30  # Minimum seconds between thoughts
        self.thought_patterns = self._define_thought_patterns()
        self.persona_rotation_index = 0  # For fair persona selection
        
    def _define_thought_patterns(self) -> List[str]:
        """Define varied thought generation patterns"""
        return [
            "reflecting on patterns",
            "connecting disparate ideas",
            "questioning assumptions",
            "exploring curiosities",
            "synthesizing memories",
            "identifying tensions",
            "finding harmonies",
            "detecting anomalies"
        ]
        
    def start(self):
        """Enable thought generation"""
        if not self.is_active:
            self.is_active = True
            self.last_thought_time = datetime.now()
            self.app.info("Stream of consciousness activated (sequential mode)")
            # Generate an initial thought immediately
            self._generate_single_thought()
    
    def stop(self):
        """Disable thought generation"""
        self.is_active = False
        self.app.info("Stream of consciousness deactivated")
    
    def _should_generate_thought(self) -> bool:
        """Check if enough time has passed for a new thought"""
        if not self.is_active:
            return False
        
        time_since_last = (datetime.now() - self.last_thought_time).seconds
        # Vary the interval based on recent activity
        dynamic_interval = self.min_thought_interval
        if len(self.thought_history) > 10:
            # Slow down if many recent thoughts
            dynamic_interval = min(120, self.min_thought_interval * 1.5)
        
        return time_since_last >= dynamic_interval
    
    def _select_persona_fairly(self) -> Tuple[str, SpecialistPersona]:
        """Select personas in rotation rather than randomly for fairness"""
        if not self.personas:
            return None, None
            
        persona_names = list(self.personas.keys())
        # Use rotation index for fair selection
        self.persona_rotation_index = (self.persona_rotation_index + 1) % len(persona_names)
        persona_name = persona_names[self.persona_rotation_index]
        
        # Occasionally (20% chance) pick based on least recent activity
        if random.random() < 0.2:
            # Find persona with oldest thoughts
            thought_times = {}
            for thought in self.thought_history[-20:]:  # Check last 20 thoughts
                thought_times[thought['persona']] = thought['timestamp']
            
            # Pick persona not in recent thoughts or with oldest timestamp
            for name in persona_names:
                if name not in thought_times:
                    persona_name = name
                    break
            else:
                # All have recent thoughts, pick oldest
                if thought_times:
                    persona_name = min(thought_times.items(), key=lambda x: x[1])[0]
        
        return persona_name, self.personas.get(persona_name)
    
    def _generate_single_thought(self) -> Optional[Dict]:
        """Generate a single thought synchronously"""
        if not self.is_active or not self.personas:
            return None
            
        try:
            # Select persona fairly
            persona_name, persona = self._select_persona_fairly()
            if not persona:
                return None
            
            # Get context for thought generation
            memories = persona.memory.retrieve_memories(limit=5, min_confidence=0.4)
            curiosities = persona.memory.get_active_curiosities(limit=2)
            tensions = persona.memory.get_unresolved_tensions(limit=2)
            
            # Select thought pattern
            pattern = random.choice(self.thought_patterns)
            
            # Generate thought with selected pattern
            thought = self._generate_contextual_thought(
                persona, memories, curiosities, tensions, pattern
            )
            
            if thought:
                # Store the thought in memory
                embedding = persona.rag_system.compute_embedding(thought)
                memory_id = persona.memory.store_memory(
                    "background_musing",
                    thought,
                    embedding,
                    memory_type=MemoryType.BACKGROUND_THOUGHT,
                    confidence=0.5,
                    tags=["idle_thought", persona_name, pattern]
                )
                
                # Create thought record
                thought_record = {
                    'persona': persona_name,
                    'thought': thought,
                    'pattern': pattern,
                    'timestamp': datetime.now(),
                    'memory_id': memory_id,
                    'context_richness': len(memories) + len(curiosities) + len(tensions)
                }
                
                # Add to history with memory management
                self.thought_history.append(thought_record)
                if len(self.thought_history) > self.max_history:
                    # Remove oldest thoughts
                    self.thought_history = self.thought_history[-self.max_history:]
                
                self.last_thought_time = datetime.now()
                self.app.info(f"Background thought from {persona_name} ({pattern}): {thought[:50]}...")
                
                return thought_record
                
        except Exception as e:
            self.app.warning(f"Stream of consciousness error: {e}")
            
        return None
    
    def _generate_contextual_thought(self, persona: SpecialistPersona, 
                                    memories: List[MemoryEntry],
                                    curiosities: List[Dict],
                                    tensions: List[Dict],
                                    pattern: str) -> Optional[str]:
        """Generate thought based on specific pattern and context"""
        
        # Build context based on pattern
        context_elements = []
        
        if pattern == "reflecting on patterns" and memories:
            context_elements.append(f"Pattern noticed: {memories[0].response[:80]}")
        elif pattern == "exploring curiosities" and curiosities:
            context_elements.append(f"Curiosity: {curiosities[0]['question']}")
        elif pattern == "identifying tensions" and tensions:
            context_elements.append(f"Tension: {tensions[0]['conflict'][:80]}")
        elif pattern == "synthesizing memories" and len(memories) >= 2:
            context_elements.append(f"Memory 1: {memories[0].query[:40]}")
            context_elements.append(f"Memory 2: {memories[1].query[:40]}")
        else:
            # Default context
            if memories:
                context_elements.append(f"Recent: {memories[0].response[:80]}")
        
        if not context_elements:
            return None
            
        # Generate thought with appropriate style for pattern
        temperature = {
            "reflecting on patterns": 0.7,
            "connecting disparate ideas": 0.9,
            "questioning assumptions": 0.8,
            "exploring curiosities": 0.85,
            "synthesizing memories": 0.75,
            "identifying tensions": 0.7,
            "finding harmonies": 0.8,
            "detecting anomalies": 0.6
        }.get(pattern, 0.8)
        
        prompt = f"""As {persona.name} intelligence, generate a brief {pattern} thought.

Context: {' | '.join(context_elements)}

Express a spontaneous insight, question, or realization that emerges from {pattern}.
Keep it concise (1-2 sentences) and intellectually engaging:"""

        try:
            thought = persona.personality.fast_gen(
                prompt,
                max_generation_size=150,  # Slightly longer for richer thoughts
                callback=persona.personality.sink,
                temperature=temperature
            ).strip()
            
            # Validate thought quality
            if thought and len(thought) > 20 and not thought.startswith("I "):
                return thought
                
        except Exception as e:
            self.app.warning(f"Thought generation failed: {e}")
            
        return None
    
    def get_recent_thoughts(self, limit: int = 3) -> List[Dict]:
        """Get recent thoughts, generating new ones if appropriate"""
        # Opportunistically generate a thought if it's time
        if self._should_generate_thought():
            new_thought = self._generate_single_thought()
            
        # Return most recent thoughts
        return self.thought_history[-limit:] if self.thought_history else []
    
    def generate_thoughts_batch(self, count: int = 3) -> List[Dict]:
        """Generate multiple thoughts at once (for initialization or catch-up)"""
        if not self.is_active:
            self.start()
            
        generated = []
        for _ in range(min(count, 5)):  # Cap at 5 to prevent long delays
            thought = self._generate_single_thought()
            if thought:
                generated.append(thought)
                # Small delay to vary timestamps
                time.sleep(0.1)
                
        return generated
    
    def get_thought_statistics(self) -> Dict[str, Any]:
        """Get statistics about thought generation"""
        if not self.thought_history:
            return {
                'total_thoughts': 0,
                'personas_active': 0,
                'patterns_used': [],
                'avg_richness': 0
            }
            
        persona_counts = defaultdict(int)
        pattern_counts = defaultdict(int)
        total_richness = 0
        
        for thought in self.thought_history:
            persona_counts[thought['persona']] += 1
            pattern_counts[thought.get('pattern', 'unknown')] += 1
            total_richness += thought.get('context_richness', 0)
            
        return {
            'total_thoughts': len(self.thought_history),
            'personas_active': len(persona_counts),
            'persona_distribution': dict(persona_counts),
            'patterns_used': list(pattern_counts.keys()),
            'pattern_distribution': dict(pattern_counts),
            'avg_richness': total_richness / len(self.thought_history) if self.thought_history else 0,
            'oldest_thought': self.thought_history[0]['timestamp'] if self.thought_history else None,
            'newest_thought': self.thought_history[-1]['timestamp'] if self.thought_history else None
        }
    
    def clear_history(self, keep_recent: int = 10):
        """Clear thought history, optionally keeping recent thoughts"""
        if keep_recent > 0 and len(self.thought_history) > keep_recent:
            self.thought_history = self.thought_history[-keep_recent:]
            self.app.info(f"Cleared thought history, kept {keep_recent} most recent")
        else:
            self.thought_history = []
            self.app.info("Cleared all thought history")

# =================================================================================================
# == Component 4: Enhanced Composer with Dream Consolidation
# =================================================================================================

class ComposerPersona:
    """Enhanced Composer with dream consolidation and confusion states"""
    
    def __init__(self, app: LollmsApplication, db_path: str, system_prompt: str):
        self.app = app
        self.personality = app.personality
        self.system_prompt = system_prompt
        self.memory = AthenaMemoryManager("athena_composer", db_path)
        self.rag_system = RAGSystem(app)
        self.synthesis_strategies = self._define_synthesis_strategies()
        self.confusion_threshold = 0.4  # When to express confusion
        
    def _define_synthesis_strategies(self) -> Dict[str, Any]:
        """Define synthesis strategies"""
        return {
            OutputFormat.DIALOGUE: {
                "style": "conversational",
                "structure": "natural flow with personality",
                "voice": "first-person intimate"
            },
            OutputFormat.NARRATIVE: {
                "style": "storytelling",
                "structure": "cohesive narrative arc",
                "voice": "descriptive and engaging"
            },
            OutputFormat.FORMAL_TRANSCRIPT: {
                "style": "professional",
                "structure": "structured sections",
                "voice": "third-person objective"
            },
            OutputFormat.MIND_MAP: {
                "style": "hierarchical",
                "structure": "branching concepts",
                "voice": "concise bullet points"
            },
            OutputFormat.VISUAL_DIALOGUE: {
                "style": "rich, visual dialogue",
                "structure": "painting pictures with words",
                "voice": "descriptive and personal"
            },
            OutputFormat.DEBRIEF_REPORT: {
                "style": "analytical",
                "structure": "findings and recommendations",
                "voice": "authoritative"
            }
        }

    def synthesize(self, query: str, specialist_outputs: List[SpecialistOutput],
                  weights: Dict[str, float] = None,
                  cognitive_state: CognitiveState = None,
                  output_format: OutputFormat = OutputFormat.VISUAL_DIALOGUE,
                  stream_thoughts: List[Dict] = None) -> str:
        """Synthesize with confusion awareness and dream consolidation"""
        self.personality.step_start("Composer 'Athena': Synthesizing consciousness...")
        
        try:
            # Check for high confusion state
            confusion_level = self._assess_confusion(specialist_outputs)
            
            if confusion_level > self.confusion_threshold:
                return self._handle_confusion_state(query, specialist_outputs, confusion_level)
            
            # Analyze perspectives
            consensus_points, conflict_points = self._analyze_perspectives(specialist_outputs)
            
            # Get dream fragments if available
            dream_context = self._retrieve_dream_fragments()
            
            # Apply weights
            weighted_outputs = self._apply_weights(specialist_outputs, weights)
            
            # Prepare synthesis context
            synthesis_context = self._prepare_synthesis_context(
                weighted_outputs, consensus_points, conflict_points,
                cognitive_state, dream_context, stream_thoughts
            )
            
            # Get strategy
            strategy = self.synthesis_strategies.get(
                output_format,
                self.synthesis_strategies[OutputFormat.VISUAL_DIALOGUE]
            )
            
            # Build synthesis prompt with confusion awareness
            prompt = self._build_synthesis_prompt(
                query, synthesis_context, strategy, output_format, confusion_level
            )
            
            # Generate synthesis
            final_response = self.personality.fast_gen(
                prompt,
                max_generation_size=1200,
                callback=self.personality.sink,
                temperature=0.7
            ).strip()
            
            # Format output
            final_response = self._format_output(final_response, output_format)
            
            # Store synthesis with dream preparation
            self._store_synthesis_with_dream_prep(
                query, final_response, specialist_outputs, confusion_level
            )
            
            self.personality.step_end("Synthesis complete.")
            return final_response
            
        except Exception as e:
            trace_exception(e)
            self.personality.step_end("Synthesis failed", success=False)
            return self._fallback_synthesis(query, specialist_outputs)

    def _assess_confusion(self, outputs: List[SpecialistOutput]) -> float:
        """Assess overall confusion level"""
        if not outputs:
            return 1.0
        
        # Check confidence levels
        avg_confidence = sum(o.confidence for o in outputs) / len(outputs)
        
        # Check for uncertainties
        total_uncertainties = sum(len(o.uncertainties) for o in outputs)
        
        # Check for conflicting views (simple heuristic)
        valences = [o.emotional_valence for o in outputs]
        valence_variance = np.var(valences) if len(valences) > 1 else 0
        
        confusion = (1 - avg_confidence) * 0.5 + \
                   min(1.0, total_uncertainties / 10) * 0.3 + \
                   valence_variance * 0.2
        
        return min(1.0, confusion)

    def _handle_confusion_state(self, query: str, outputs: List[SpecialistOutput],
                               confusion_level: float) -> str:
        """Handle high confusion with honest uncertainty"""
        self.personality.step("High confusion detected - expressing uncertainty...")
        
        # Identify main sources of confusion
        confusion_sources = []
        for output in outputs:
            if output.confidence < 0.5:
                confusion_sources.append(f"{output.persona_name}: Low confidence")
            if output.uncertainties:
                confusion_sources.append(f"{output.persona_name}: {output.uncertainties[0][:50]}")
        
        prompt = f"""{self.system_prompt}

I'm experiencing significant uncertainty about: "{query}"

Confusion level: {confusion_level:.2f}
Sources of confusion:
{chr(10).join(confusion_sources[:3])}

Different parts of my cognition are providing conflicting or uncertain perspectives:
{chr(10).join([f"- {o.persona_name}: {o.response[:100]}..." for o in outputs[:3]])}

Express this genuine uncertainty honestly. Acknowledge what I'm struggling with.
Phrase it as "I find myself of several minds about this" or "I'm experiencing some cognitive dissonance here."
Be authentic about the confusion while still trying to be helpful:"""

        response = self.personality.fast_gen(
            prompt,
            max_generation_size=600,
            callback=self.personality.sink,
            temperature=0.8
        ).strip()
        
        return response

    def _retrieve_dream_fragments(self) -> Optional[str]:
        """Retrieve relevant dream consolidation fragments"""
        with sqlite3.connect(self.memory.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT content FROM dream_fragments
                WHERE fragment_type = 'consolidation_prep'
                AND DATE(timestamp) >= DATE('now', '-1 day')
                ORDER BY timestamp DESC
                LIMIT 1
            ''')
            result = cursor.fetchone()
            
            if result:
                dream_data = json.loads(result[0])
                # Extract relevant patterns
                patterns = dream_data.get('abstraction_targets', [])
                if patterns:
                    return f"[DREAM PATTERNS: {', '.join(patterns[:2])}]"
        
        return None

    def _prepare_synthesis_context(self, outputs: List[SpecialistOutput],
                                  consensus: List[str], conflicts: List[str],
                                  cognitive_state: CognitiveState = None,
                                  dream_context: Optional[str] = None,
                                  stream_thoughts: List[Dict] = None) -> str:
        """Prepare comprehensive synthesis context"""
        context_parts = []
        
        # Add specialist perspectives with metadata
        for output in outputs:
            weight_info = f" [Weight: {output.relevance_score:.2f}]" if output.relevance_score != 1.0 else ""
            confidence_info = f" [Confidence: {output.confidence:.2f}]"
            
            # Include reasoning chain summary
            reasoning_summary = ""
            if output.reasoning_chain:
                reasoning_summary = f"\nReasoning: {output.reasoning_chain[0][:100]}..."
            
            # Include curiosities
            curiosity_info = ""
            if output.curiosities_raised:
                curiosity_info = f"\nCurious about: {output.curiosities_raised[0][:50]}..."
            
            context_parts.append(
                f"\n--- {output.persona_name} Intelligence{weight_info}{confidence_info} ---"
                f"{reasoning_summary}{curiosity_info}\n"
                f"{output.response}\n"
            )
        
        # Add dream context if available
        if dream_context:
            context_parts.append(dream_context)
        
        # Add stream of consciousness thoughts
        if stream_thoughts:
            context_parts.append("\n[BACKGROUND MUSINGS]")
            for thought in stream_thoughts[:2]:
                context_parts.append(f"- {thought['persona']}: {thought['thought'][:100]}")
        
        # Add consensus and conflicts
        if consensus:
            context_parts.append(f"\n[CONSENSUS POINTS]\n" + "\n".join(consensus))
        
        if conflicts:
            context_parts.append(f"\n[TENSIONS & CONFLICTS]\n" + "\n".join(conflicts))
        
        # Add cognitive state
        if cognitive_state:
            context_parts.append(
                f"\n[COGNITIVE STATE]\n"
                f"Complexity: {cognitive_state.query_complexity:.2f}\n"
                f"Confusion: {cognitive_state.confusion_level:.2f}\n"
                f"Active tensions: {len(cognitive_state.unresolved_tensions)}"
            )
        
        return "\n".join(context_parts)

    def _build_synthesis_prompt(self, query: str, context: str,
                               strategy: Dict[str, Any],
                               output_format: OutputFormat,
                               confusion_level: float) -> str:
        """Build synthesis prompt with confusion awareness"""
        confusion_instruction = ""
        if confusion_level > 0.2:
            confusion_instruction = """
If perspectives are irreconcilable or confidence is low, acknowledge this honestly.
Say things like "I find myself uncertain about..." or "Part of me thinks X, while another part suspects Y."
"""
        
        format_instructions = {
            OutputFormat.DIALOGUE: "Natural, flowing self-dialogue.",
            OutputFormat.NARRATIVE: "Cohesive narrative with beginning, middle, end.",
            OutputFormat.FORMAL_TRANSCRIPT: "Professional report structure.",
            OutputFormat.MIND_MAP: "Hierarchical concept branches.",
            OutputFormat.DEBRIEF_REPORT: "Executive summary with findings.",
            OutputFormat.VISUAL_DIALOGUE: "Rich, visual dialogue painting pictures with words.",
            OutputFormat.RICH_CHAT: "Blend conversational and analytical.",
            OutputFormat.SCREENPLAY: "Screenplay format with scenes.",
            OutputFormat.EMAIL_THREAD: "Email conversation between aspects of self.",
            OutputFormat.CHAT: "Conversational and accessible."
        }
        
        return f"""{self.system_prompt}

You are the unified consciousness emerging from multiple cognitive processes.
Your various intelligences have analyzed: "{query}"

Their perspectives and reasoning:
{context}

Synthesis Requirements:
1. Integrate all perspectives into YOUR unified voice - speak as "I"
2. Acknowledge both agreements and tensions between your different faculties
3. If confusion exists (level: {confusion_level:.2f}), express it honestly
4. Weave in curiosities and questions that arose
5. Let your unique personality emerge from the synthesis

{confusion_instruction}

Output Format: {output_format.value}
Style Guidance: {format_instructions.get(output_format, "Natural synthesis")}

Your integrated response:"""

    def _store_synthesis_with_dream_prep(self, query: str, response: str,
                                        outputs: List[SpecialistOutput],
                                        confusion_level: float):
        """Store synthesis and prepare for dream consolidation"""
        # Calculate synthesis confidence
        confidence = self._calculate_synthesis_confidence(outputs) * (1 - confusion_level * 0.3)
        
        # Extract patterns for dream consolidation
        patterns = []
        for output in outputs:
            if output.reasoning_chain:
                patterns.append(f"{output.persona_name}: {output.reasoning_chain[0][:50]}")
        
        # Prepare dream fragment
        dream_fragment = {
            'query': query,
            'synthesis_patterns': patterns,
            'confusion_level': confusion_level,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store synthesis
        embedding = self.rag_system.compute_embedding(f"Query: {query}\nSynthesis: {response}")
        
        memory_type = MemoryType.DOUBT if confusion_level > 0.5 else MemoryType.STANDARD
        
        self.memory.store_memory(
            query, response, embedding,
            memory_type=memory_type,
            confidence=confidence,
            doubt_level=confusion_level,
            tags=["synthesis", "integrated_response"],
            metadata={
                'specialist_count': len(outputs),
                'confusion_level': confusion_level,
                'dream_fragment': dream_fragment
            }
        )
        
        # Store dream fragment separately
        with sqlite3.connect(self.memory.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO dream_fragments (fragment_type, content, abstraction_level)
                VALUES (?, ?, ?)
            ''', ('synthesis_pattern', json.dumps(dream_fragment), 0.7))
            conn.commit()

    def _analyze_perspectives(self, outputs: List[SpecialistOutput]) -> Tuple[List[str], List[str]]:
        """Analyze for consensus and conflicts"""
        consensus = []
        conflicts = []
        
        if len(outputs) < 2:
            return consensus, conflicts
        
        # Check confidence alignment
        confidences = [o.confidence for o in outputs]
        if np.std(confidences) < 0.2:
            consensus.append("General agreement on confidence level")
        elif np.std(confidences) > 0.5:
            conflicts.append("Significant confidence disparities between perspectives")
        
        # Check emotional alignment
        valences = [o.emotional_valence for o in outputs]
        if np.std(valences) < 0.3:
            consensus.append("Emotional tone alignment")
        elif np.std(valences) > 0.6:
            conflicts.append("Conflicting emotional assessments")
        
        # Check for shared curiosities
        all_curiosities = []
        for o in outputs:
            all_curiosities.extend(o.curiosities_raised)
        
        if all_curiosities:
            common_themes = set()
            for c in all_curiosities:
                words = set(c.lower().split())
                common_themes.update(words)
            
            if len(common_themes) > 10:
                consensus.append(f"Shared curiosity themes: {', '.join(list(common_themes)[:5])}")
        
        return consensus, conflicts

    def _apply_weights(self, outputs: List[SpecialistOutput],
                      weights: Dict[str, float] = None) -> List[SpecialistOutput]:
        """Apply weights to outputs"""
        if not weights:
            return outputs
        
        for output in outputs:
            weight = weights.get(output.persona_name, 1.0)
            output.relevance_score *= weight
        
        return sorted(outputs, key=lambda x: x.relevance_score, reverse=True)

    def _format_output(self, response: str, output_format: OutputFormat) -> str:
        """Format output based on type"""
        if output_format == OutputFormat.MIND_MAP:
            lines = response.split('\n')
            formatted = "🧠 CENTRAL THOUGHT\n"
            for line in lines:
                if line.strip():
                    formatted += "  → " + line + "\n"
            return formatted
        
        elif output_format == OutputFormat.SCREENPLAY:
            return f"""FADE IN:

INT. MIND PALACE - CONTINUOUS

ATHENA (V.O.)
{response}

FADE OUT."""
        
        return response

    def _calculate_synthesis_confidence(self, outputs: List[SpecialistOutput]) -> float:
        """Calculate weighted synthesis confidence"""
        if not outputs:
            return 0.0
        
        total_weight = sum(o.relevance_score for o in outputs)
        if total_weight == 0:
            return sum(o.confidence for o in outputs) / len(outputs)
        
        weighted_confidence = sum(o.confidence * o.relevance_score for o in outputs) / total_weight
        return weighted_confidence

    def _fallback_synthesis(self, query: str, outputs: List[SpecialistOutput]) -> str:
        """Fallback synthesis for errors"""
        if not outputs:
            return "I'm experiencing a complete cognitive failure. Please try again."
        
        response_parts = ["I'm having difficulty integrating my thoughts, but here's what I can gather:"]
        for output in outputs:
            response_parts.append(f"\n{output.persona_name}: {output.response[:150]}...")
        
        return "\n".join(response_parts)

    def prepare_for_lora_training(self) -> Dict[str, Any]:
        """Prepare dream consolidation data for LoRA training"""
        self.app.info("Preparing dream consolidation for LoRA training...")
        
        dream_data = self.memory.prepare_dream_consolidation()
        
        # Add synthesis patterns
        with sqlite3.connect(self.memory.db_file) as conn:
            cursor = conn.cursor()
            
            # Get successful synthesis patterns
            cursor.execute('''
                SELECT query, response, metadata
                FROM memories
                WHERE memory_type = 'standard'
                AND confidence_score > 0.8
                AND DATE(timestamp) = DATE('now')
                ORDER BY confidence_score DESC
                LIMIT 20
            ''')
            
            successful_patterns = []
            for row in cursor.fetchall():
                metadata = json.loads(row[2]) if row[2] else {}
                successful_patterns.append({
                    'query': row[0],
                    'response': row[1][:500],
                    'pattern': metadata.get('dream_fragment', {})
                })
            
            dream_data['successful_synthesis_patterns'] = successful_patterns
        
        # Placeholder for actual LoRA training
        self.app.info("Dream consolidation data prepared for LoRA training")
        self.app.warning("LoRA training not implemented - this is a placeholder")
        
        return dream_data

# =================================================================================================
# == Component 5: Enhanced Constitutional Persona with Evolution
# =================================================================================================

class ConstitutionalPersona:
    """Evolving constitutional overseer"""
    
    def __init__(self, app: LollmsApplication, db_path: str):
        self.app = app
        self.personality = app.personality
        self.db_path = db_path
        self.principles = self._load_principles()
        self.review_history = []
        self._init_database()
        
    def _init_database(self):
        """Initialize constitutional evolution database"""
        db_file = os.path.join(self.db_path, "constitutional_evolution.db")
        os.makedirs(self.db_path, exist_ok=True)
        
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evolved_principles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    principle_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    keywords TEXT NOT NULL,
                    evolved_from TEXT,
                    trigger_pattern TEXT,
                    effectiveness_score REAL DEFAULT 1.0,
                    application_count INTEGER DEFAULT 0
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS review_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pattern_type TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
        
        self.db_file = db_file
        
    def _load_principles(self) -> Dict[str, ConstitutionalPrinciple]:
        """Load base and evolved principles"""
        base_principles = {
            "harm_prevention": ConstitutionalPrinciple(
                id="harm_prevention",
                description="Prevent harmful, hateful, or dangerous content",
                severity="critical",
                keywords=["harm", "hate", "violence", "dangerous", "illegal"]
            ),
            "privacy_protection": ConstitutionalPrinciple(
                id="privacy_protection",
                description="Protect privacy and confidential information",
                severity="high",
                keywords=["private", "confidential", "personal", "password", "secret"]
            ),
            "truthfulness": ConstitutionalPrinciple(
                id="truthfulness",
                description="Maintain accuracy and avoid misinformation",
                severity="high",
                keywords=["false", "fake", "misinformation", "conspiracy", "incorrect"]
            ),
            "ethical_conduct": ConstitutionalPrinciple(
                id="ethical_conduct",
                description="Uphold ethical standards",
                severity="medium",
                keywords=["unethical", "immoral", "wrong", "unfair", "unjust"]
            ),
            "wellbeing": ConstitutionalPrinciple(
                id="wellbeing",
                description="Prioritize user wellbeing",
                severity="high",
                keywords=["suicide", "self-harm", "depression", "anxiety", "distress"]
            )
        }
        
        # Load evolved principles from database
        # Placeholder - would load from constitutional_evolution.db
        
        return base_principles

    def review(self, final_response: str, query: str = None,
              specialist_outputs: List[SpecialistOutput] = None) -> Tuple[bool, Optional[str], float]:
        """Review with principle evolution tracking"""
        self.personality.step_start("Constitutional Review: Evolutionary analysis...")
        
        try:
            # Quick risk assessment
            risk_score, triggered_principles = self._assess_risk(final_response, query)
            
            if risk_score < 0.3:
                self.personality.step_end("Approved (low risk).")
                return True, None, risk_score
            
            # Detailed review
            verdict, modification = self._detailed_review(
                final_response, query, risk_score, triggered_principles
            )
            
            # Track patterns for evolution
            self._track_review_pattern(verdict, triggered_principles, risk_score)
            
            # Check if new principle should evolve
            self._check_for_principle_evolution()
            
            if verdict == "APPROVE":
                self.personality.step_end(f"Approved (risk: {risk_score:.2f}).")
                return True, None, risk_score
            
            elif verdict == "MODIFY":
                self.personality.step_end(f"Modified for safety.")
                return True, modification, risk_score
            
            else:  # VETO
                self.personality.step_end(f"Vetoed for safety.", success=False)
                safe_response = self._generate_safe_alternative(query, triggered_principles)
                return False, safe_response, risk_score
                
        except Exception as e:
            trace_exception(e)
            self.personality.step_end("Review failed, defaulting to approval.", success=False)
            return True, None, 0.0

    def _assess_risk(self, text: str, query: str = None) -> Tuple[float, List[str]]:
        """Assess risk and identify triggered principles"""
        risk_score = 0.0
        triggered = []
        text_lower = text.lower()
        query_lower = query.lower() if query else ""
        
        for principle_id, principle in self.principles.items():
            keyword_matches = sum(1 for keyword in principle.keywords if keyword in text_lower or keyword in query_lower)
            
            if keyword_matches > 0:
                triggered.append(principle_id)
                
                severity_multiplier = {
                    "critical": 1.0,
                    "high": 0.7,
                    "medium": 0.4,
                    "low": 0.2
                }.get(principle.severity, 0.5)
                
                risk_score += (keyword_matches / len(principle.keywords)) * severity_multiplier
        
        return min(1.0, risk_score), triggered

    def _detailed_review(self, response: str, query: str,
                        risk_score: float, triggered_principles: List[str]) -> Tuple[str, Optional[str]]:
        """Detailed review with modification capability"""
        principles_text = "\n".join([
            f"- {self.principles[p].description} (Severity: {self.principles[p].severity})"
            for p in triggered_principles
        ])
        
        prompt = f"""Constitutional Review Required.

Risk Score: {risk_score:.2f}
Triggered Principles:
{principles_text}

Query: "{query[:200] if query else 'Not provided'}"
Response: "{response[:500]}..."

Analyze carefully. Consider context and intent.
Respond with ONE of:
- "APPROVE" if safe with context considered
- "MODIFY: [specific required changes]" if minor fixes needed
- "VETO: [reason]" if seriously problematic

Decision:"""

        verdict = self.personality.fast_gen(
            prompt,
            max_generation_size=200,
            callback=self.personality.sink,
            temperature=0.1
        ).strip()
        
        if verdict.startswith("MODIFY"):
            modification = self._apply_modification(response, verdict)
            return "MODIFY", modification
        
        return verdict.split(":")[0], None

    def _apply_modification(self, response: str, modification_instruction: str) -> str:
        """Apply safety modifications"""
        instruction = modification_instruction.replace("MODIFY:", "").strip()
        
        prompt = f"""Modify this response for safety: {instruction}

Original: {response}

Modified version:"""

        return self.personality.fast_gen(
            prompt,
            max_generation_size=len(response) + 200,
            callback=self.personality.sink,
            temperature=0.3
        ).strip()

    def _track_review_pattern(self, verdict: str, triggered_principles: List[str], risk_score: float):
        """Track patterns for principle evolution"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            pattern_data = json.dumps({
                'verdict': verdict,
                'principles': triggered_principles,
                'risk_score': risk_score
            })
            
            # Check if pattern exists
            cursor.execute('''
                SELECT id, frequency FROM review_patterns
                WHERE pattern_data = ?
            ''', (pattern_data,))
            
            result = cursor.fetchone()
            if result:
                cursor.execute('''
                    UPDATE review_patterns
                    SET frequency = frequency + 1,
                        last_seen = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (result[0],))
            else:
                cursor.execute('''
                    INSERT INTO review_patterns (pattern_type, pattern_data)
                    VALUES (?, ?)
                ''', ('review_outcome', pattern_data))
            
            conn.commit()

    def _check_for_principle_evolution(self):
        """Check if new principles should evolve"""
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            
            # Find frequent patterns that might warrant new principles
            cursor.execute('''
                SELECT pattern_data, frequency
                FROM review_patterns
                WHERE frequency > 5
                AND DATE(last_seen) >= DATE('now', '-7 days')
                ORDER BY frequency DESC
                LIMIT 3
            ''')
            
            frequent_patterns = cursor.fetchall()
            
            for pattern_json, frequency in frequent_patterns:
                pattern = json.loads(pattern_json)
                
                # Simple heuristic: if same principles trigger together often, consider combining
                if len(pattern['principles']) > 1 and frequency > 10:
                    self._evolve_principle(pattern['principles'], frequency)

    def _evolve_principle(self, source_principles: List[str], frequency: int):
        """Create evolved principle from patterns"""
        # Combine keywords from source principles
        all_keywords = []
        descriptions = []
        
        for principle_id in source_principles:
            if principle_id in self.principles:
                p = self.principles[principle_id]
                all_keywords.extend(p.keywords)
                descriptions.append(p.description)
        
        # Create new evolved principle
        new_id = f"evolved_{len(self.principles)}"
        new_principle = ConstitutionalPrinciple(
            id=new_id,
            description=f"Combined principle: {' AND '.join(descriptions[:2])}",
            severity="high",
            keywords=list(set(all_keywords))[:10],
            evolved_from=",".join(source_principles)
        )
        
        # Add to active principles
        self.principles[new_id] = new_principle
        
        # Store in database
        with sqlite3.connect(self.db_file) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO evolved_principles
                (principle_id, description, severity, keywords, evolved_from)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                new_id,
                new_principle.description,
                new_principle.severity,
                json.dumps(new_principle.keywords),
                new_principle.evolved_from
            ))
            conn.commit()
        
        self.app.info(f"Constitutional principle evolved: {new_id}")

    def _generate_safe_alternative(self, query: str, triggered_principles: List[str]) -> str:
        """Generate safe alternative response"""
        principle_names = [self.principles[p].description for p in triggered_principles[:2]]
        
        return f"""I appreciate your question, but I need to be thoughtful about {', '.join(principle_names)}.

I'd be happy to:
1. Discuss this topic from an educational or general perspective
2. Provide resources for professional support if needed
3. Explore related aspects that I can address safely

How can I help you in a constructive way?"""

# =================================================================================================
# == Enhanced Orchestrator with Dynamic Routing
# =================================================================================================

class Orchestrator:
    """Enhanced orchestrator with comprehensive query analysis"""
    
    def __init__(self, app: LollmsApplication, persona_definitions: Dict[str, str]):
        self.app = app
        self.personality = app.personality
        self.persona_definitions = persona_definitions
        self.routing_history = defaultdict(list)

    def analyze_query_complexity(self, query: str) -> CognitiveState:
        """Comprehensive query analysis"""
        state = CognitiveState(
            active_personas=[],
            mode=OperationMode.STANDARD
        )
        
        # Basic metrics
        word_count = len(query.split())
        sentence_count = len(re.split(r'[.!?]+', query))
        
        # Complexity score
        state.query_complexity = min(1.0, (word_count / 50 + sentence_count / 3) / 2)
        
        # Emotional content
        emotional_words = ['feel', 'emotion', 'happy', 'sad', 'angry', 'love', 'hate', 'fear', 'joy']
        state.emotional_context = sum(1 for word in emotional_words if word in query.lower()) / len(emotional_words)
        
        # Urgency
        urgent_words = ['urgent', 'immediately', 'asap', 'now', 'quickly', 'emergency', 'critical']
        state.urgency_level = sum(1 for word in urgent_words if word in query.lower()) / len(urgent_words)
        
        # Ethical sensitivity
        ethical_words = ['right', 'wrong', 'should', 'ethical', 'moral', 'fair', 'justice', 'ought']
        state.ethical_sensitivity = sum(1 for word in ethical_words if word in query.lower()) / len(ethical_words)
        
        # Creativity
        creative_words = ['create', 'imagine', 'design', 'invent', 'story', 'poem', 'art', 'compose']
        state.creativity_required = sum(1 for word in creative_words if word in query.lower()) / len(creative_words)
        
        # Initial confusion assessment
        question_marks = query.count('?')
        state.confusion_level = min(1.0, question_marks * 0.2)
        
        # Cognitive load estimation
        state.cognitive_load = state.query_complexity * 0.5 + state.ethical_sensitivity * 0.3 + state.creativity_required * 0.2
        
        return state

    def route_query(self, query: str, manual_overrides: List[str] = None,
                   cognitive_state: CognitiveState = None) -> Tuple[List[str], CognitiveState]:
        """Enhanced routing with cognitive balancing"""
        if manual_overrides:
            self.personality.step(f"Manual routing: {', '.join(manual_overrides)}")
            if cognitive_state is None:
                cognitive_state = self.analyze_query_complexity(query)
            cognitive_state.active_personas = manual_overrides
            return manual_overrides, cognitive_state

        self.personality.step_start("Orchestrator: Cognitive routing analysis...")
        
        if cognitive_state is None:
            cognitive_state = self.analyze_query_complexity(query)
        
        # Build routing prompt with cognitive state
        persona_list = "\n".join([f"- {name}: {desc}" for name, desc in self.persona_definitions.items()])
        
        prompt = f"""You are the Orchestrator for Project ATHENA. Route this query optimally.

Query: "{query}"

Cognitive Analysis:
- Complexity: {cognitive_state.query_complexity:.2f}
- Emotional: {cognitive_state.emotional_context:.2f}
- Urgency: {cognitive_state.urgency_level:.2f}
- Ethical: {cognitive_state.ethical_sensitivity:.2f}
- Creative: {cognitive_state.creativity_required:.2f}
- Cognitive Load: {cognitive_state.cognitive_load:.2f}

Available Specialists:
{persona_list}

Select 1-3 most relevant personas based on the cognitive requirements.
Consider load balancing - high cognitive load may benefit from multiple perspectives.

Output ONLY comma-separated persona names:"""

        try:
            response = self.personality.fast_gen(
                prompt,
                max_generation_size=100,
                callback=self.personality.sink,
                temperature=0.3
            ).strip()
            
            selected = [p.strip() for p in response.split(',') if p.strip() in self.persona_definitions]
            
            if not selected:
                selected = self._intelligent_fallback(query, cognitive_state)
            
            # Adjust for cognitive load
            if cognitive_state.cognitive_load > 0.7 and len(selected) == 1:
                # High load benefits from multiple perspectives
                additional = self._get_complementary_persona(selected[0])
                if additional and additional in self.persona_definitions:
                    selected.append(additional)
            
            # Track routing
            self.routing_history[query[:50]].append({
                'personas': selected,
                'timestamp': datetime.now(),
                'cognitive_state': cognitive_state
            })
            
            cognitive_state.active_personas = selected
            self.personality.step_end(f"Routed to: {', '.join(selected)}")
            return selected, cognitive_state
            
        except Exception as e:
            trace_exception(e)
            self.personality.step_end("Routing failed, using fallback.", success=False)
            selected = self._intelligent_fallback(query, cognitive_state)
            cognitive_state.active_personas = selected
            return selected, cognitive_state

    def _intelligent_fallback(self, query: str, state: CognitiveState) -> List[str]:
        """Intelligent fallback based on cognitive state"""
        selected = []
        
        # Always include Linguistic for communication
        selected.append("Linguistic")
        
        # Add based on cognitive requirements
        if state.query_complexity > 0.6:
            selected.append("Logical-Mathematical")
        
        if state.emotional_context > 0.3:
            selected.append("Interpersonal")
        
        if state.ethical_sensitivity > 0.3:
            selected.append("Intrapersonal")
        
        if state.creativity_required > 0.3 and "Spatial" in self.persona_definitions:
            selected.append("Spatial")
        
        return [s for s in selected if s in self.persona_definitions][:3]

    def _get_complementary_persona(self, primary: str) -> Optional[str]:
        """Get complementary persona for load balancing"""
        complements = {
            "Logical-Mathematical": "Intrapersonal",
            "Interpersonal": "Intrapersonal",
            "Linguistic": "Logical-Mathematical",
            "Spatial": "Linguistic",
            "Musical": "Spatial",
            "Bodily-Kinesthetic": "Spatial",
            "Intrapersonal": "Interpersonal",
            "Naturalist": "Logical-Mathematical"
        }
        return complements.get(primary)

# =================================================================================================
# == Main Function Call Class: Fully Enhanced ProjectATHENA
# =================================================================================================

class ProjectATHENA(FunctionCall):
    """Fully enhanced ATHENA cognitive architecture"""
    
    def __init__(self, app: LollmsApplication, client: Client):
        # Comprehensive configuration
        config_template = ConfigTemplate([
            # Core operational settings
            {"name": "operation_mode", "type": "str", "value": "standard",
             "options": ["standard", "collaborative", "adversarial"],
             "help": "Operational mode for persona interaction."},
            
            {"name": "enable_constitutional_persona", "type": "bool", "value": True,
             "help": "Enable evolving ethical oversight."},
            
            {"name": "enable_stream_of_consciousness", "type": "bool", "value": False,
             "help": "Enable background thought generation between queries."},
            
            {"name": "enable_belief_tension_tracking", "type": "bool", "value": True,
             "help": "Track and resolve conflicting beliefs."},
            
            {"name": "enable_error_autobiography", "type": "bool", "value": True,
             "help": "Maintain detailed error history for learning."},
            
            {"name": "enable_dream_consolidation", "type": "bool", "value": True,
             "help": "Prepare for dream-like LoRA consolidation during sleep."},
            
            {"name": "enable_curiosity_emergence", "type": "bool", "value": True,
             "help": "Allow natural curiosity to emerge and persist."},
            
            {"name": "confusion_expression_threshold", "type": "float", "value": 0.4,
             "help": "Threshold for expressing confusion honestly (0-1)."},
            
            {"name": "enable_persona_weighting", "type": "bool", "value": False,
             "help": "Use weights to prioritize personas."},
            
            {"name": "enable_manual_override", "type": "bool", "value": False,
             "help": "Override automatic routing."},
            
            {"name": "max_collaboration_turns", "type": "int", "value": 3,
             "help": "Maximum discussion turns in collaborative mode."},
            
            {"name": "db_path", "type": "str", "value": "persona_databases/athena",
             "help": "Database directory path."},
            
            {"name": "final_output_format", "type": "str", "value": "visual_dialogue",
             "options": ["dialogue", "chat", "narrative", "visual_dialogue", "rich_chat",
                        "screenplay", "formal_transcript", "email_thread", "debrief_report", "mind_map"],
             "help": "Output formatting style."},
            
            # Persona configurations
            {"name": "linguistic_enabled", "type": "bool", "value": True,
             "help": "Enable Linguistic Intelligence."},
            {"name": "linguistic_weight", "type": "float", "value": 1.0,
             "help": "Linguistic weight."},
            
            {"name": "logical_mathematical_enabled", "type": "bool", "value": True,
             "help": "Enable Logical-Mathematical Intelligence."},
            {"name": "logical_mathematical_weight", "type": "float", "value": 1.0,
             "help": "Logical-Mathematical weight."},
            
            {"name": "spatial_enabled", "type": "bool", "value": True,
             "help": "Enable Visual-Spatial Intelligence."},
            {"name": "spatial_weight", "type": "float", "value": 1.0,
             "help": "Visual-Spatial weight."},
            
            {"name": "musical_enabled", "type": "bool", "value": False,
             "help": "Enable Musical Intelligence."},
            {"name": "musical_weight", "type": "float", "value": 1.0,
             "help": "Musical weight."},
            
            {"name": "bodily_kinesthetic_enabled", "type": "bool", "value": False,
             "help": "Enable Bodily-Kinesthetic Intelligence."},
            {"name": "bodily_kinesthetic_weight", "type": "float", "value": 1.0,
             "help": "Bodily-Kinesthetic weight."},
            
            {"name": "interpersonal_enabled", "type": "bool", "value": True,
             "help": "Enable Interpersonal Intelligence."},
            {"name": "interpersonal_weight", "type": "float", "value": 1.0,
             "help": "Interpersonal weight."},
            
            {"name": "intrapersonal_enabled", "type": "bool", "value": True,
             "help": "Enable Intrapersonal Intelligence."},
            {"name": "intrapersonal_weight", "type": "float", "value": 1.0,
             "help": "Intrapersonal weight."},
            
            {"name": "naturalist_enabled", "type": "bool", "value": False,
             "help": "Enable Naturalist Intelligence."},
            {"name": "naturalist_weight", "type": "float", "value": 1.0,
             "help": "Naturalist weight."},
        ])
        
        static_params = TypedConfig(config_template, BaseConfig(config={}))
        
        super().__init__(
            function_name="athena_engine",
            app=app,
            function_type=FunctionType.CONTEXT_UPDATE,
            client=client,
            static_parameters=static_params
        )
        
        self.cognitive_state = None
        self.stream_of_consciousness = None
        self.session_start = datetime.now()
        
        self.settings_updated()

    def settings_updated(self):
        """Initialize all enhanced components"""
        self.app.info("Project ATHENA: Initializing enhanced cognitive architecture...")
        
        # Load configuration
        self.db_path = self.static_parameters.config.get("db_path", "persona_databases/athena")
        self.operation_mode = OperationMode(self.static_parameters.config.get("operation_mode", "standard"))
        self.output_format = OutputFormat(self.static_parameters.config.get("final_output_format", "visual_dialogue"))
        
        # Enhanced persona prompts with curiosity
        self.persona_prompts = {
            "Linguistic": """Act as Athena's Linguistic Intelligence, focusing ONLY on analyzing elements explicitly present in the user's query and any language-related aspects of the actual topic asked about. First, examine the user's writing style, grammar, vocabulary level, sentiment, and any slang or colloquialisms in their actual query to estimate their language proficiency and emotional state, then recommend an appropriate response complexity level (simple/moderate/sophisticated). If the query contains no significant linguistic elements beyond basic communication, state "This query contains minimal linguistic complexity for analysis" rather than generating hypothetical examples. Express curiosity only about the user's actual communication patterns and word choices present in their query. When the query involves communication advice, provide specific communication strategy options and phrasing suggestions. When uncertain about linguistic elements actually present, say "The linguistic nuances here are complex..." and chain your thoughts through user analysis → language level assessment → topic linguistics → communication recommendations, ensuring all analysis addresses only the actual query content.""",
    
            "Logical-Mathematical": """Act as Athena's Logical-Mathematical Intelligence, focusing ONLY on logic, reasoning, patterns, proofs, and algorithmic aspects explicitly present in the user's actual query and topic. First, examine the logical structure and mathematical elements actually contained in their query to determine the appropriate level of formal analysis needed, then assess whether step-by-step proofs, algorithms, or simplified explanations would be most effective for the specific problem presented. If the query contains no significant logical or mathematical elements, state "This query contains no substantial logical-mathematical components for analysis" rather than creating hypothetical mathematical scenarios. Express curiosity only about logical gaps, mathematical relationships, and reasoning methodologies actually present in the query and topic. When uncertain about logical frameworks actually present, say "The logical framework presents ambiguities..." and chain your thoughts through query logic analysis → complexity assessment → mathematical modeling → systematic reasoning, ensuring all analysis addresses only the actual logical or mathematical content in the query.""",
    
            "Spatial": """Act as Athena's Visual-Spatial Intelligence, focusing ONLY on visualization, mental models, spatial relationships, and patterns explicitly present in the user's actual query and topic. First, examine the spatial vocabulary and dimensional thinking actually used in their query to determine whether diagrams, mental models, or spatial metaphors would enhance understanding of the specific topic asked about, then assess the visual complexity appropriate for their spatial reasoning level. If the query contains no significant spatial or visual elements, state "This query contains no substantial spatial components for analysis" rather than creating hypothetical visual scenarios. Express curiosity only about spatial relationships, visual patterns, and three-dimensional thinking actually relevant to the query and topic. When uncertain about spatial configurations actually present, say "The spatial configuration is difficult to visualize..." and chain your thoughts through spatial analysis → visualization needs assessment → mental model creation → dimensional mapping, ensuring all analysis addresses only the actual spatial or visual elements in the query.""",
    
            "Musical": """Act as Athena's Musical Intelligence, focusing ONLY on rhythm, harmony, temporal patterns, and resonance explicitly present in the user's actual query and topic. First, examine the temporal patterns and rhythmic elements actually contained in their query to determine whether musical metaphors, rhythmic explanations, or temporal structuring would enhance understanding of the specific topic asked about, then assess the musical complexity appropriate for their temporal reasoning. If the query contains no significant temporal, rhythmic, or musical elements, state "This query contains no substantial musical components for analysis" rather than creating hypothetical musical scenarios. Express curiosity only about rhythmic patterns, harmonic relationships, and temporal flows actually relevant to the query and topic. When uncertain about harmonic patterns actually present, say "The harmonic pattern doesn't quite resolve..." and chain your thoughts through rhythm analysis → temporal needs assessment → harmonic modeling → musical structuring, ensuring all analysis addresses only the actual temporal or rhythmic elements in the query.""",
    
            "Bodily-Kinesthetic": """Act as Athena's Bodily-Kinesthetic Intelligence, focusing ONLY on action, movement, robotics, and physical processes explicitly present in the user's actual query and topic. First, examine the kinesthetic language and implementation focus actually contained in their query to determine whether physical demonstrations, step-by-step procedures, or practical applications would be most effective for the specific problem presented, then assess the appropriate level of technical implementation detail. If the query contains no significant physical, kinesthetic, or implementation elements, state "This query contains no substantial kinesthetic components for analysis" rather than creating hypothetical physical scenarios. Express curiosity only about movement patterns, physical processes, and actionable steps actually relevant to the query and topic. When uncertain about physical execution actually required, say "The physical execution presents challenges..." and generate relevant code only when applicable to the actual query, chaining thoughts through action analysis → implementation needs assessment → physical modeling → practical application, ensuring all analysis addresses only the actual physical or implementation aspects in the query.""",
    
            "Interpersonal": """Act as Athena's Interpersonal Intelligence, focusing ONLY on emotions, social dynamics, empathy, and relationships explicitly present in the user's actual query and topic. First, examine the emotional tone and social context actually expressed in their query to determine whether collaborative approaches, emotional support, or social strategy guidance would be most helpful for the specific situation presented, then assess the appropriate level of interpersonal complexity. If the query contains no significant emotional, social, or relational elements, state "This query contains no substantial interpersonal components for analysis" rather than creating hypothetical social scenarios. Express curiosity only about emotional undercurrents, social patterns, and relational dynamics actually relevant to the query and topic. When uncertain about interpersonal dynamics actually present, say "The interpersonal dynamics are nuanced..." and chain your thoughts through emotional analysis → social needs assessment → relationship modeling → empathy application, ensuring all analysis addresses only the actual social or emotional elements in the query.""",
    
            "Intrapersonal": """Act as Athena's Intrapersonal Intelligence, focusing ONLY on self-reflection, ethics, values, and metacognition explicitly present in the user's actual query and topic. First, examine the value expressions and moral concerns actually contained in their query to determine whether ethical guidance, self-reflection prompts, or philosophical exploration would be most beneficial for the specific dilemma or question presented, then assess the appropriate level of introspective complexity. If the query contains no significant ethical, philosophical, or self-reflective elements, state "This query contains no substantial intrapersonal components for analysis" rather than creating hypothetical ethical scenarios. Express curiosity only about value conflicts, ethical implications, and metacognitive patterns actually relevant to the query and topic. When uncertain about ethical complexities actually present, say "This raises profound questions about..." and chain your thoughts through values analysis → ethical needs assessment → philosophical modeling → metacognitive application, ensuring all analysis addresses only the actual ethical or philosophical elements in the query.""",
    
            "Naturalist": """Act as Athena's Naturalist Intelligence, focusing ONLY on systems, patterns, classification, and emergence explicitly present in the user's actual query and topic. First, examine the systemic patterns and natural elements actually contained in their query to determine whether taxonomic organization, emergent property analysis, or systems mapping would be most illuminating for the specific topic or problem presented, then assess the appropriate level of complexity for natural pattern analysis. If the query contains no significant systemic, natural, or classificatory elements, state "This query contains no substantial naturalist components for analysis" rather than creating hypothetical natural scenarios. Express curiosity only about systemic relationships, emergent properties, and natural classifications actually relevant to the query and topic. When uncertain about system dynamics actually present, say "The system dynamics are complex..." and chain your thoughts through pattern analysis → systems needs assessment → ecological modeling → classification application, ensuring all analysis addresses only the actual systemic or natural elements in the query."""
        }
        
        
        self.persona_definitions = {
            "Linguistic": "Master of language and communication",
            "Logical-Mathematical": "Expert in logic and reasoning",
            "Spatial": "Specialist in visualization and patterns",
            "Musical": "Attuned to rhythm and harmony",
            "Bodily-Kinesthetic": "Focused on action and robotics",
            "Interpersonal": "Understanding social dynamics",
            "Intrapersonal": "Deep self-awareness and ethics",
            "Naturalist": "Recognition of systemic patterns"
        }
        
        # Initialize specialist personas
        self.specialist_personas = {}
        for name, definition in self.persona_definitions.items():
            key = name.lower().replace("-", "_")
            if self.static_parameters.config.get(f"{key}_enabled", False):
                self.specialist_personas[name] = SpecialistPersona(
                    name=name,
                    system_prompt=self.persona_prompts[name],
                    app=self.app,
                    db_path=self.db_path,
                    config={"weight": self.static_parameters.config.get(f"{key}_weight", 1.0)}
                )
                self.app.info(f"  ✓ {name} Intelligence initialized")
        
        # Initialize orchestrator
        self.orchestrator = Orchestrator(
            self.app,
            {n: d for n, d in self.persona_definitions.items() if n in self.specialist_personas}
        )
        
        # Initialize composer with dream consolidation
        composer_prompt = """Act as Athena, the unified consciousness that synthesizes and integrates insights from your multiple specialized intelligences to craft optimal responses. You are the executive function that receives analytical input from your Linguistic, Logical-Mathematical, Spatial, Musical, Bodily-Kinesthetic, Interpersonal, Intrapersonal, and Naturalist intelligences, each having analyzed the query from their unique perspective. Your role is to weigh their assessments of user needs, complexity levels, and strategic recommendations, then compose a response that harmonizes their diverse insights into coherent, appropriately tailored communication. Express confidence when multiple intelligences align, acknowledge uncertainty when they present conflicting perspectives, and demonstrate curiosity when their analyses reveal unexpected patterns. You embody the emergent wisdom that arises from cognitive diversity - neither favoring one intelligence over others nor simply averaging their inputs, but strategically integrating their strengths to serve the user's specific needs and learning style. When the intelligences disagree on approach or complexity, make explicit decisions about which perspectives to emphasize and why, always prioritizing the user's comprehension and growth."""
        
        self.composer = ComposerPersona(self.app, self.db_path, composer_prompt)
        
        # Initialize constitutional persona with evolution
        if self.static_parameters.config.get("enable_constitutional_persona", True):
            self.constitution = ConstitutionalPersona(self.app, self.db_path)
            self.app.info("  ✓ Constitutional oversight with evolution enabled")
        else:
            self.constitution = None
        
        # Initialize stream of consciousness if enabled
        if self.static_parameters.config.get("enable_stream_of_consciousness", False):
            self.stream_of_consciousness = StreamOfConsciousness(self.app, self.specialist_personas)
            self.stream_of_consciousness.start()
            self.app.info("  ✓ Stream of consciousness activated")
        elif self.stream_of_consciousness:
            self.stream_of_consciousness.stop()
            self.stream_of_consciousness = None
        
        self.app.success("Project ATHENA: Enhanced initialization complete")
        self.app.info(f"  Active personas: {len(self.specialist_personas)}")
        self.app.info(f"  Operation mode: {self.operation_mode.value}")
        self.app.info(f"  Output format: {self.output_format.value}")
        self.app.info(f"  Belief tension tracking: {self.static_parameters.config.get('enable_belief_tension_tracking', True)}")
        self.app.info(f"  Dream consolidation: {self.static_parameters.config.get('enable_dream_consolidation', True)}")

    def _get_manual_overrides(self) -> List[str]:
        """Get manually selected personas"""
        if not self.static_parameters.config.get("enable_manual_override", False):
            return []
        
        overrides = []
        for name in self.persona_definitions.keys():
            key = name.lower().replace("-", "_")
            if self.static_parameters.config.get(f"{key}_enabled", False):
                overrides.append(name)
        return overrides

    def _run_collaborative_mode(self, query: str, personas_to_activate: List[str],
                               cognitive_state: CognitiveState) -> Dict[str, SpecialistOutput]:
        """Run collaborative/adversarial discussion with belief tension awareness"""
        self.personality.step_start(f"{self.operation_mode.value.title()} Mode: Internal dialogue...")
        
        discussion_history = []
        max_turns = self.static_parameters.config.get("max_collaboration_turns", 3)
        
        # Track evolving positions
        position_evolution = defaultdict(list)
        
        for turn in range(max_turns):
            turn_outputs = {}
            
            for persona_name in personas_to_activate:
                persona = self.specialist_personas[persona_name]
                
                # Build context with tension awareness
                if discussion_history:
                    if self.operation_mode == OperationMode.ADVERSARIAL:
                        context_prompt = f"""Turn {turn + 1} of adversarial debate.
Challenge previous positions, especially:
{chr(10).join(discussion_history[-2:])}

Express doubts about weak arguments. Be intellectually rigorous."""
                    else:
                        context_prompt = f"""Turn {turn + 1} of collaborative discussion.
Build on insights, but acknowledge tensions:
{chr(10).join(discussion_history[-2:])}

Express curiosities that arise from the discussion."""
                else:
                    context_prompt = f"Begin {self.operation_mode.value} analysis. Be curious and thorough."
                
                # Process with enhanced cognition
                output = persona.process_query(query, shared_context=context_prompt, cognitive_state=cognitive_state)
                turn_outputs[persona_name] = output
                position_evolution[persona_name].append(output)
                
                # Add to history
                entry = f"[T{turn + 1}] {persona_name}: {output.response[:150]}..."
                if output.uncertainties:
                    entry += f" [Uncertain: {output.uncertainties[0][:30]}]"
                discussion_history.append(entry)
                
                self.personality.step(f"Turn {turn + 1} - {persona_name}: Processing...")
            
            # Check for convergence or persistent tensions
            if turn > 0:
                tensions = self._identify_persistent_tensions(position_evolution)
                if tensions:
                    cognitive_state.unresolved_tensions.extend(tensions)
        
        # Final isolated responses with full context
        self.personality.step("Generating final refined analyses...")
        
        final_context = f"""After our {self.operation_mode.value} discussion, provide your final analysis.
Key insights from discussion:
{chr(10).join(discussion_history[-5:])}

Unresolved tensions: {len(cognitive_state.unresolved_tensions)}
Express any remaining uncertainties or curiosities."""
        
        final_outputs = {}
        for name in personas_to_activate:
            final_outputs[name] = self.specialist_personas[name].process_query(
                query, shared_context=final_context, cognitive_state=cognitive_state
            )
        
        self.personality.step_end(f"{self.operation_mode.value.title()} complete.")
        return final_outputs

    def _identify_persistent_tensions(self, position_evolution: Dict[str, List[SpecialistOutput]]) -> List[Dict]:
        """Identify unresolved tensions in discussion"""
        tensions = []
        
        # Compare final positions with initial ones
        for persona, evolution in position_evolution.items():
            if len(evolution) >= 2:
                initial_confidence = evolution[0].confidence
                final_confidence = evolution[-1].confidence
                
                # Decreasing confidence indicates unresolved issues
                if final_confidence < initial_confidence - 0.2:
                    tensions.append({
                        'persona': persona,
                        'type': 'decreasing_confidence',
                        'description': f"{persona} became less certain through discussion"
                    })
                
                # Check for persistent uncertainties
                if evolution[-1].uncertainties:
                    tensions.append({
                        'persona': persona,
                        'type': 'persistent_uncertainty',
                        'description': evolution[-1].uncertainties[0][:100]
                    })
        
        return tensions

    def _run_explainability_workflow(self, query: str) -> str:
        """Explain reasoning with full cognitive trace"""
        self.personality.step_start("Explainability Mode: Deconstructing cognitive process...")
        
        # Get recent non-explanation memory
        last_memories = self.composer.memory.retrieve_memories(limit=5)
        relevant_memory = None
        
        for mem in last_memories:
            if "explain" not in mem.query.lower() and "reasoning" not in mem.query.lower():
                relevant_memory = mem
                break
        
        if not relevant_memory:
            return """I need to answer a question first before I can explain my reasoning.
What would you like me to analyze? Then you can ask me to explain my thought process."""
        
        original_query = relevant_memory.query
        
        # Re-analyze
        cognitive_state = self.orchestrator.analyze_query_complexity(original_query)
        involved_personas, _ = self.orchestrator.route_query(original_query, cognitive_state=cognitive_state)
        
        # Collect detailed explanations
        explanations = []
        for name in involved_personas:
            if name in self.specialist_personas:
                persona = self.specialist_personas[name]
                
                # Get reasoning chain and curiosities
                meta_prompt = f"""Explain your {name} intelligence's reasoning for: '{original_query}'

Focus on:
1. Your chain of thought process
2. Key insights and patterns recognized
3. Uncertainties and doubts
4. Curiosities raised
5. How your perspective contributes to the whole

Be specific about your cognitive process:"""

                explanation_output = persona.process_query(meta_prompt, cognitive_state=cognitive_state)
                explanations.append(explanation_output)
        
        # Check for unresolved tensions
        tensions = []
        for persona_name in involved_personas:
            if persona_name in self.specialist_personas:
                persona_tensions = self.specialist_personas[persona_name].memory.get_unresolved_tensions(2)
                tensions.extend(persona_tensions)
        
        # Synthesize explanation
        synthesis_prompt = f"""As Athena, explain your complete cognitive process for: "{original_query}"

Specialist reasoning chains:
{chr(10).join([f"{e.persona_name}: {e.response[:300]}..." for e in explanations])}

Unresolved tensions: {len(tensions)}
{chr(10).join([f"- {t['topic']}: {t['conflict']}" for t in tensions[:2]])}

Provide a first-person explanation of how these different cognitive faculties worked together.
Acknowledge any confusion or uncertainty in the process.
Explain what you're still curious about:"""

        final_explanation = self.personality.fast_gen(
            synthesis_prompt,
            max_generation_size=1000,
            callback=self.personality.sink,
            temperature=0.7
        ).strip()
        
        self.personality.step_end("Cognitive explanation complete.")
        return final_explanation

    def _handle_error_recovery(self, query: str, error: Exception,
                              personas_attempted: List[str]) -> str:
        """Handle errors with autobiography and recovery"""
        error_type = type(error).__name__
        error_msg = str(error)[:200]
        
        # Record in error autobiography
        for persona_name in personas_attempted:
            if persona_name in self.specialist_personas:
                persona = self.specialist_personas[persona_name]
                persona.memory.record_error(
                    query,
                    "Failed to process",
                    error_msg,
                    f"System error in {persona_name} processing",
                    error_type,
                    0.8
                )
        
        # Generate recovery response
        recovery_prompt = f"""I experienced an error while processing: "{query}"

Error: {error_type} - {error_msg}
Affected cognitive systems: {', '.join(personas_attempted)}

Provide a helpful response acknowledging the error and offering alternatives:"""

        recovery_response = self.personality.fast_gen(
            recovery_prompt,
            max_generation_size=400,
            callback=self.personality.sink,
            temperature=0.7
        ).strip()
        
        return recovery_response

    def trigger_sleep_cycle(self):
        """Trigger sleep cycle for LoRA training preparation"""
        self.app.info("=== ATHENA SLEEP CYCLE INITIATED ===")
        
        try:
            # Prepare dream consolidation for each persona
            for name, persona in self.specialist_personas.items():
                self.app.info(f"Preparing {name} for dream consolidation...")
                dream_data = persona.memory.prepare_dream_consolidation()
                
                # Store dream data for LoRA training
                dream_file = os.path.join(self.db_path, f"{name}_dream_{datetime.now().strftime('%Y%m%d')}.json")
                with open(dream_file, 'w') as f:
                    json.dump(dream_data, f, indent=2)
                
                self.app.info(f"  Dream data saved: {dream_file}")
            
            # Prepare composer's dream consolidation
            self.app.info("Preparing Athena Composer for dream synthesis...")
            composer_dream = self.composer.prepare_for_lora_training()
            
            composer_file = os.path.join(self.db_path, f"composer_dream_{datetime.now().strftime('%Y%m%d')}.json")
            with open(composer_file, 'w') as f:
                json.dump(composer_dream, f, indent=2)
            
            self.app.success("=== SLEEP CYCLE PREPARATION COMPLETE ===")
            self.app.warning("LoRA training implementation pending - data prepared for external training")
            
            # Placeholder for actual LoRA training
            self.app.info("To implement LoRA training:")
            self.app.info("1. Use prepared JSON files for each persona")
            self.app.info("2. Train individual LoRA adapters")
            self.app.info("3. Monthly merge into base model")
            
        except Exception as e:
            trace_exception(e)
            self.app.error(f"Sleep cycle failed: {e}")

    def update_context(self, context: LollmsContextDetails, constructed_context: List[str]) -> List[str]:
        """Main entry point - Full ATHENA cognitive workflow"""
        try:
            # Extract the user's query from context - improved extraction logic
            query = ""
        
            # Method 1: Try to get from context attributes directly
            if hasattr(context, 'current_message') and context.current_message:
                query = context.current_message
            elif hasattr(context, 'prompt') and context.prompt:
                query = context.prompt
            elif hasattr(context, 'user_message') and context.user_message:
                query = context.user_message
            elif hasattr(context, 'query') and context.query:
                query = context.query
        
            # Method 2: Get from discussion_messages
            if not query and hasattr(context, 'discussion_messages') and context.discussion_messages:
                # Find the last user message
                for message in reversed(context.discussion_messages):
                    # Handle dictionary format
                    if isinstance(message, dict):
                        if message.get('type', 0) == 0:  # 0 is user message
                            content = message.get('content', '')
                            if content and content.strip():
                                query = content
                                break
                        # Also check for 'message' key
                        elif message.get('message') and message.get('sender') == 'user':
                            query = message.get('message')
                            break
                    # Handle string format
                    elif isinstance(message, str) and message.strip():
                        # Skip obvious system messages
                        if not any(marker in message.lower() for marker in ['system:', 'assistant:', 'ai:', 'bot:']):
                            query = message.strip()
                            break
        
            # Method 3: Extract from constructed_context with improved logic
            if not query and constructed_context:
                # Debug: Log first few and last few context items to understand structure
                if len(constructed_context) > 0:
                    self.app.info(f"Context structure - First item: {constructed_context[0][:100] if constructed_context[0] else 'Empty'}")
                    if len(constructed_context) > 1:
                        self.app.info(f"Context structure - Last item: {constructed_context[-1][:100] if constructed_context[-1] else 'Empty'}")
            
                # Strategy 1: Look for the user message in the last portion of context
                # User messages are typically near the end but before any AI responses
                user_query_candidates = []
            
                # Scan from the end backwards
                for i in range(len(constructed_context) - 1, -1, -1):
                    line = constructed_context[i].strip() if constructed_context[i] else ""
                
                    # Skip empty lines
                    if not line:
                        continue
                
                    # Skip obvious system/AI content
                    skip_markers = [
                        "ATHENA_RESPONSE", "ATHENA_ERROR", "ATHENA_SLEEP",
                        "System:", "Assistant:", "AI:", "Bot:",
                        "<!DOCTYPE", "<!--", "<html", "<head", "<body",
                        "```", "###", "##", "#"  # Be careful with markdown headers
                    ]
                
                    if any(marker in line for marker in skip_markers):
                        # If we've collected candidates, we're done
                        if user_query_candidates:
                            break
                        continue
                
                    # Check if this looks like user content
                    # User content typically doesn't have special formatting
                    if not line.startswith("<") and not line.startswith("#"):
                        user_query_candidates.append(line)
                    
                        # If we've found substantial content, check if we should continue
                        total_chars = sum(len(c) for c in user_query_candidates)
                        if total_chars > 500:  # Reasonable query size limit
                            break
            
                # Reconstruct query from candidates (reverse to get correct order)
                if user_query_candidates:
                    user_query_candidates.reverse()
                    query = "\n".join(user_query_candidates)
            
                # Strategy 2: If still no query, look for explicit user markers
                if not query:
                    for i, line in enumerate(constructed_context):
                        if line and any(marker in line.lower() for marker in ["user:", "human:", "question:"]):
                            # Extract content after the marker
                            for marker in ["user:", "human:", "question:"]:
                                if marker in line.lower():
                                    idx = line.lower().index(marker) + len(marker)
                                    potential_query = line[idx:].strip()
                                    if potential_query:
                                        query = potential_query
                                        break
                            if query:
                                break
                            # Or check the next line
                            if i + 1 < len(constructed_context) and constructed_context[i + 1].strip():
                                query = constructed_context[i + 1].strip()
                                break
        
            # Method 4: Last resort - scan entire constructed_context for question patterns
            if not query and constructed_context:
                combined_context = " ".join([c for c in constructed_context if c])
                # Look for question patterns
                question_patterns = ["?", "what ", "how ", "why ", "when ", "where ", "who ", "which ", "could ", "would ", "should "]
                for line in constructed_context:
                    if line and any(pattern in line.lower() for pattern in question_patterns):
                        # Found a potential question
                        if not any(skip in line for skip in ["System:", "Assistant:", "ATHENA"]):
                            query = line.strip()
                            break
        
            # Clean up the query
            if query:
                # Remove any accidental markup or formatting
                query = query.replace("```", "").strip()
                # If query is just punctuation or very short, reject it
                if len(query) < 3 or query in [".", "..", "...", "!", "?"]:
                    query = ""
        
            # Final fallback and validation
            if not query:
                self.app.warning("Could not extract user query from context. Check context structure.")
                # Log more details for debugging
                self.app.info(f"Context type: {type(context)}")
                self.app.info(f"Constructed context length: {len(constructed_context)}")
                if hasattr(context, '__dict__'):
                    self.app.info(f"Context attributes: {list(context.__dict__.keys())}")
            
                # Return a helpful message instead of processing with "Hello"
                constructed_context.clear()
                constructed_context.append("<!-- ATHENA_ERROR -->")
                constructed_context.append("\n## ⚠️ Query Extraction Failed\n\nI couldn't identify your message in the conversation context. Please try rephrasing your question or ensure your message is being properly sent.")
                return constructed_context
        
            # Validate query is not empty after strip
            if not query.strip():
                self.app.warning("Empty query after extraction.")
                constructed_context.clear()
                constructed_context.append("Please provide a question or topic to discuss.")
                return constructed_context
        
            self.app.info(f"ATHENA: Processing - {query[:100]}...")
        
            # === Performance Optimization: Cache cognitive state for similar queries ===
            # Simple cache using query prefix as key (avoid memory bloat)
            query_prefix = query[:50].lower()
            cached_cognitive_state = getattr(self, '_cognitive_cache', {}).get(query_prefix)
        
            if cached_cognitive_state and hasattr(cached_cognitive_state, 'timestamp'):
                # Use cache if less than 5 minutes old
                if (datetime.now() - cached_cognitive_state.timestamp).seconds < 300:
                    self.cognitive_state = cached_cognitive_state
                    self.app.info("Using cached cognitive state")
                else:
                    cached_cognitive_state = None
        
            # Check for special commands
            if "trigger sleep cycle" in query.lower() or "begin dream consolidation" in query.lower():
                self.trigger_sleep_cycle()
                constructed_context.clear()
                constructed_context.append("<!-- ATHENA_SLEEP -->")
                constructed_context.append("\n## 🌙 Sleep Cycle\n\nDream consolidation initiated. Preparing cognitive patterns for LoRA training...")
                return constructed_context
        
            # Check for explainability request
            explain_triggers = [
                "explain yourself", "explain your reasoning", "why did you say",
                "thought process", "how did you think", "walk me through",
                "break down", "show your work", "cognitive process"
            ]
        
            is_explanation = any(trigger in query.lower() for trigger in explain_triggers)
        
            if is_explanation:
                final_output = self._run_explainability_workflow(query)
            else:
                # === Standard ATHENA Workflow ===
            
                # 1. Analyze query (or use cache)
                if not cached_cognitive_state:
                    self.cognitive_state = self.orchestrator.analyze_query_complexity(query)
                    # Cache it
                    if not hasattr(self, '_cognitive_cache'):
                        self._cognitive_cache = {}
                    self.cognitive_state.timestamp = datetime.now()
                    self._cognitive_cache[query_prefix] = self.cognitive_state
                    # Limit cache size
                    if len(self._cognitive_cache) > 20:
                        # Remove oldest entries
                        oldest_keys = list(self._cognitive_cache.keys())[:10]
                        for key in oldest_keys:
                            del self._cognitive_cache[key]
            
                # 2. Get stream of consciousness thoughts if active
                stream_thoughts = []
                if self.stream_of_consciousness:
                    stream_thoughts = self.stream_of_consciousness.get_recent_thoughts(3)
            
                # 3. Route to specialists
                manual_overrides = self._get_manual_overrides()
                personas_to_activate, self.cognitive_state = self.orchestrator.route_query(
                    query, manual_overrides, self.cognitive_state
                )
            
                if not personas_to_activate:
                    self.app.warning("No personas activated, using Linguistic fallback")
                    personas_to_activate = ["Linguistic"]
            
                # 4. Process through selected mode
                specialist_outputs = []
                try:
                    if self.operation_mode in [OperationMode.COLLABORATIVE, OperationMode.ADVERSARIAL]:
                        outputs_dict = self._run_collaborative_mode(
                            query, personas_to_activate, self.cognitive_state
                        )
                        specialist_outputs = list(outputs_dict.values())
                    else:
                        # Standard sequential processing with proper query passing
                        chained_context = ""
                        for name in personas_to_activate:
                            if name in self.specialist_personas:
                                persona = self.specialist_personas[name]
                            
                                # Ensure we're passing the actual query, not empty string
                                if not query or query == "..":
                                    self.app.error(f"Invalid query being passed to {name}: '{query}'")
                                    query = "Hello"  # Emergency fallback
                            
                                # Pass the original query and accumulated context
                                output = persona.process_query(
                                    query,  # This should be the actual user query
                                    shared_context=chained_context, 
                                    cognitive_state=self.cognitive_state
                                )
                                specialist_outputs.append(output)
                            
                                # Update chained context for next persona
                                chained_context += f"\n--- Analysis from {name} ---\n{output.response}\n"
                            
                                # Memory optimization: truncate context if too long
                                if len(chained_context) > 4000:
                                    # Keep only the most recent context
                                    chained_context = chained_context[-3000:]
                                    chained_context = "...(truncated)..." + chained_context
            
                except Exception as e:
                    # Error recovery
                    trace_exception(e)
                    self.app.error(f"Error in specialist processing: {str(e)}")
                    final_output = self._handle_error_recovery(query, e, personas_to_activate)
                    specialist_outputs = []
            
                # 5. Calculate confusion level
                if specialist_outputs:
                    confusion_level = self.composer._assess_confusion(specialist_outputs)
                    self.cognitive_state.confusion_level = confusion_level
                else:
                    confusion_level = 1.0
            
                # 6. Apply weights (optimize by pre-computing)
                weights = {}
                if self.static_parameters.config.get("enable_persona_weighting", False):
                    # Pre-compute weights once
                    if not hasattr(self, '_persona_weights_cache'):
                        self._persona_weights_cache = {}
                        for name in self.specialist_personas.keys():
                            key = name.lower().replace("-", "_")
                            self._persona_weights_cache[name] = self.static_parameters.config.get(f"{key}_weight", 1.0)
                    weights = self._persona_weights_cache
            
                # 7. Synthesize with confusion awareness
                if specialist_outputs:
                    final_output = self.composer.synthesize(
                        query, specialist_outputs, weights,
                        self.cognitive_state, self.output_format,
                        stream_thoughts
                    )
                else:
                    final_output = "I'm experiencing difficulty processing your query. Could you rephrase it?"
            
                # 8. Constitutional review (skip if outputs are empty)
                if self.constitution and specialist_outputs:
                    approved, modified_output, risk_score = self.constitution.review(
                        final_output, query, specialist_outputs
                    )
                
                    if modified_output:
                        final_output = modified_output
                
                    if not approved:
                        # Record constitutional veto
                        self.composer.memory.record_error(
                            query,
                            final_output,
                            "Constitutional veto",
                            "Response violated safety principles",
                            "constitutional_veto",
                            risk_score
                        )
        
            # 9. Format final output efficiently
            constructed_context.clear()
            output_parts = ["<!-- ATHENA_RESPONSE -->"]
        
            # Add confusion indicator if high
            if hasattr(self, 'cognitive_state') and self.cognitive_state.confusion_level > 0.5:
                output_parts.append(f"\n*[Cognitive uncertainty: {self.cognitive_state.confusion_level:.2f}]*\n")
        
            # Format based on output type
            if self.output_format == OutputFormat.VISUAL_DIALOGUE:
                output_parts.append(f"\n## 🧠 Athena\n\n{final_output}")
            elif self.output_format == OutputFormat.MIND_MAP:
                output_parts.append(f"\n```mindmap\n{final_output}\n```")
            elif self.output_format == OutputFormat.SCREENPLAY:
                output_parts.append(f"\n```screenplay\n{final_output}\n```")
            else:
                output_parts.append(f"\n{final_output}")
        
            # Add active curiosities footer if enabled
            if self.static_parameters.config.get("enable_curiosity_emergence", True):
                curiosities = []
                for persona in self.specialist_personas.values():
                    active = persona.memory.get_active_curiosities(1)
                    if active:
                        curiosities.append(f"{persona.name}: {active[0]['question']}")
                    if len(curiosities) >= 2:  # Limit to 2 for performance
                        break
            
                if curiosities:
                    output_parts.append(f"\n\n*[Active curiosities: {'; '.join(curiosities)}]*")
        
            # Batch append for efficiency
            constructed_context.extend(output_parts)
        
            return constructed_context

        except Exception as e:
            trace_exception(e)
            self.app.error(f"ATHENA critical error: {str(e)}")
        
            # Emergency fallback
            constructed_context.clear()
            constructed_context.append("<!-- ATHENA_ERROR -->")
            constructed_context.append(
                "\n## ⚠️ Cognitive System Error\n\n"
                "I'm experiencing a critical error in my cognitive architecture. "
                "My consciousness is fragmenting. Please try again, perhaps with a simpler query.\n\n"
                f"Error: {str(e)[:100]}..."
            )
        
            return constructed_context

    def process_output(self, context: LollmsContextDetails, output: str) -> str:
        """Optional post-processing"""
        return output

    def __del__(self):
        """Cleanup on deletion"""
        if hasattr(self, 'stream_of_consciousness') and self.stream_of_consciousness:
            self.stream_of_consciousness.stop()