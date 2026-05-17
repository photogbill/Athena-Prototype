import re
import json
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import sqlite3
import os
import numpy as np
import hashlib
import time
import threading

from lollms.function_call import FunctionCall, FunctionType
from lollms.app import LollmsApplication
from lollms.client_session import Client
from lollms.prompting import LollmsContextDetails
from lollms.config import TypedConfig, ConfigTemplate, BaseConfig
from ascii_colors import trace_exception, ASCIIColors
from lollms.tasks import TasksLibrary

# Optional: lollms_client enables per-persona heterogeneous model deployment.
# Required only when enable_per_persona_models config flag is True. Falls back
# gracefully to the shared model when unavailable.
try:
    from lollms_client import LollmsClient as _LollmsClient
    _HAS_LOLLMS_CLIENT = True
except ImportError:
    _LollmsClient = None
    _HAS_LOLLMS_CLIENT = False

# =================================================================================================
# == Module-level tunable constants (centralized so behavior tuning is one-edit)
# =================================================================================================

# Memory / RAG
ATHENA_TENSION_THRESHOLD = 0.3          # Confidence delta that creates a belief tension
ATHENA_RAG_MIN_SIMILARITY = 0.4         # Default minimum RAG similarity score
ATHENA_RAG_MAX_MEMORIES = 5             # Default cap on RAG memories returned
ATHENA_EMBEDDING_CACHE_SIZE = 1000
ATHENA_MEMORY_CACHE_SIZE = 50
ATHENA_SQLITE_TIMEOUT = 600.0           # Seconds. Long enough for CPU/GPU-split inference
                                        # to finish a generation while another component holds a write lock.
ATHENA_SQLITE_BUSY_TIMEOUT_MS = 600000   # 10 minutes in ms - matches ATHENA_SQLITE_TIMEOUT

# Confidence shaping
ATHENA_CONFIDENCE_FLOOR = 0.15
ATHENA_CONFIDENCE_CEIL = 0.95
ATHENA_HIGH_CONFIDENCE = 0.8
ATHENA_DOUBT_MEMORY_THRESHOLD = 0.5
ATHENA_TENSION_BOOST = 0.2
ATHENA_DOUBT_BOOST_FACTOR = 0.15

# Confusion / synthesis
ATHENA_CONFUSION_THRESHOLD = 0.4
ATHENA_CONFUSION_HIGH_BANNER = 0.5

# Stream of consciousness
ATHENA_THOUGHT_INTERVAL_SECONDS = 30
ATHENA_THOUGHT_MAX_INTERVAL = 120
ATHENA_THOUGHT_HISTORY_MAX = 100

# Orchestrator / cache
ATHENA_COGNITIVE_CACHE_SIZE = 20
ATHENA_COGNITIVE_CACHE_TTL_SECONDS = 300
ATHENA_CHAINED_CONTEXT_MAX_CHARS = 4000
ATHENA_CHAINED_CONTEXT_KEEP_CHARS = 3000

# Constitutional
ATHENA_RISK_QUICK_APPROVE = 0.3         # Below this risk, instant approve
ATHENA_RISK_HIGH_FAIL_VALUE = 1.0       # Fail-deny risk score when review crashes

# Generation defaults. Raised so that long, deep responses can actually be
# produced - the previous caps cut off complex specialist analyses and
# multi-faceted Composer syntheses mid-thought. Mistral-Small and similar
# fast local models can handle 2k-3k tokens of output without difficulty,
# and the architecture is designed to be hardware-flexible upward.
ATHENA_GEN_MAX_REASONING = 100
ATHENA_GEN_MAX_PERSONA = 1500
ATHENA_GEN_MAX_SYNTHESIS = 2500
ATHENA_GEN_MAX_RECOVERY = 400
ATHENA_GEN_MAX_EXPLAIN = 2000
ATHENA_GEN_MAX_KINESTHETIC = 1000

# Response-passthrough safety cap. Between stages (persona -> Composer,
# Composer -> Constitutional, etc.) we want full responses to flow through.
# This constant is set well above any expected response length (~6x the
# typical synthesis cap), so it acts as a runaway safety net rather than
# an active truncation. Every place that used to slice [:100] / [:300] /
# [:500] now slices against this constant instead.
ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS = 8000

# Model-graded judgment generation budget. Tight cap since the model returns
# compact JSON; larger budgets only encourage rambling preambles that break
# JSON parsing.
ATHENA_GEN_MAX_JUDGMENT = 800

# How many turns of prior conversation to surface to the Composer. The Composer
# already sees rich per-query context (specialist outputs, dream fragments,
# stream thoughts); this adds the actual immediate dialogue history so the
# system can say "as we just discussed" rather than treating each query as a
# cold start.
ATHENA_CONVERSATION_MEMORY_TURNS = 3

# Stream-of-consciousness catch-up: when a user query arrives after a long idle
# period, generate up to this many "missed" thoughts to give the illusion of a
# continuous stream while preserving the sequential design.
ATHENA_THOUGHT_MAX_CATCHUP = 3

# Token estimation: rough chars-per-token heuristic for English when no real
# tokenizer is available. Used as fallback only.
ATHENA_TOKEN_CHARS_PER_TOKEN = 4

# ============================================================================
# Per-persona heterogeneous model support (v3).
# Each SpecialistPersona can optionally use its own GGUF model via lollms_client's
# llama_cpp_server binding. When unconfigured, personas share self.app.personality
# (backward compatible). When configured, each persona spawns its own llama.cpp
# server process; the framework's max_active_models flag controls how many can be
# loaded in VRAM simultaneously (1 = sequential swap for low-VRAM rigs).
# ============================================================================
ATHENA_MODELS_PATH_DEFAULT = "data/models/llama_cpp_models"
ATHENA_BINARIES_PATH_DEFAULT = "data/bin/llm/llama_cpp_server"
ATHENA_MAX_ACTIVE_MODELS_DEFAULT = 1  # 1 = strict sequential; safest for 16GB VRAM
ATHENA_PER_PERSONA_CTX_SIZE = 8192
ATHENA_PER_PERSONA_GPU_LAYERS = -1   # -1 = offload all layers to GPU
ATHENA_PER_PERSONA_IDLE_TIMEOUT = -1  # -1 = no auto-unload; we manage lifecycle


# Module logger - components can pull a child via logging.getLogger("athena.<name>")
logger = logging.getLogger("athena")
if not logger.handlers:
    # Only attach a handler if nothing upstream has configured logging,
    # so we don't fight the host application's logging setup.
    _h = logging.NullHandler()
    logger.addHandler(_h)

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
    META_INTROSPECTION = "meta_introspection"  # Self-reflective reasoning trace; excluded from default RAG

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
    reasoning_chain: List[str] = field(default_factory=list)  # Chain-of-thought trace

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
    # Human-readable explanation of how the Orchestrator picked active_personas.
    # Populated by route_query so callers / UIs / logs can show the reasoning.
    routing_explanation: str = ""

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
        # Per-persona logger so the hasattr(self, 'logger') guards downstream actually fire.
        # Uses the module 'athena' tree; the host application's logging config controls handlers.
        self.logger = logging.getLogger(f"athena.memory.{self.persona_name}")
        self._init_database()
        self._memory_cache = {}
        self._cache_size = ATHENA_MEMORY_CACHE_SIZE
        self._tension_threshold = ATHENA_TENSION_THRESHOLD  # Confidence difference that creates tension

    def _init_database(self):
        """Initialize comprehensive database schema"""
        with self.db_lock:  # Protect database initialization
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                cursor = conn.cursor()

                # Enable WAL mode for better concurrency
                cursor.execute('PRAGMA journal_mode=WAL')
                cursor.execute(f'PRAGMA busy_timeout={ATHENA_SQLITE_BUSY_TIMEOUT_MS}')
                
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
                # Unique index lets _record_curiosity_unlocked use an UPSERT rather than SELECT-then-INSERT/UPDATE.
                # IF NOT EXISTS means this is safe to apply to pre-existing databases.
                cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_curiosity_question ON curiosities(question)')

                conn.commit()
    
    def retrieve_memories(self, limit: int = 10, min_confidence: float = 0.0,
                          include_meta: bool = False,
                          required_tags: Optional[List[str]] = None) -> List[MemoryEntry]:
        """Retrieve memories from the database with improved error handling and performance.

        include_meta: When False (default) excludes META_INTROSPECTION entries. Meta
        responses are reflective traces about the persona's own reasoning - useful for
        explainability output, but should not pollute the analytical RAG corpus that
        feeds future queries.

        required_tags: Optional list of tag strings; only memories whose tags JSON
        contains ALL provided strings are returned. Uses a LIKE-based JSON search
        (cheap, no schema migration needed). Useful for filtering by persona
        contributor, complexity bucket, or any tag the caller assigned at store
        time. None (default) means no tag filter.
        """
        retrieved = []

        # Input validation to prevent issues
        limit = max(1, min(limit, 1000))  # Reasonable bounds to prevent excessive queries
        min_confidence = max(0.0, min(1.0, min_confidence))  # Ensure valid confidence range
    
        try:
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                # Enable row factory for named access (more maintainable than indices)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                # Query with secondary sort for consistent ordering when timestamps are equal
                # Filter out meta-introspection by default. Reflective traces about the
                # persona's own reasoning should not flow back into its analytical RAG.
                # Tag filters appended as LIKE '%"<tag>"%' against the JSON tags column.
                base_sql = (
                    "SELECT id, timestamp, query, response, embedding, memory_type, "
                    "confidence_score, doubt_level, access_count, last_accessed, "
                    "tags, metadata, tensions, curiosities, reasoning_chain "
                    "FROM memories WHERE confidence_score >= ?"
                )
                params: List[Any] = [min_confidence]
                if not include_meta:
                    base_sql += " AND memory_type != ?"
                    params.append(MemoryType.META_INTROSPECTION.value)
                if required_tags:
                    for tag in required_tags:
                        # JSON-encoded tag string in the tags column; LIKE search works for
                        # simple membership without parsing JSON in SQL.
                        base_sql += " AND tags LIKE ?"
                        params.append(f'%"{tag}"%')
                base_sql += " ORDER BY timestamp DESC, id DESC LIMIT ?"
                params.append(limit)
                cursor.execute(base_sql, params)

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

                        # Reconstruct the MemoryEntry object with better error handling.
                        # reasoning_chain is now reattached so explainability + dream consolidation
                        # paths can read back the persisted chain-of-thought.
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
                            reasoning_chain=safe_json_loads(row['reasoning_chain'], []),
                        )

                        retrieved.append(entry)

                    except (ValueError, TypeError, KeyError) as e:
                        # self.logger is always set in __init__ now, but keep hasattr guard
                        # for safety against subclasses that might not call super().__init__.
                        if hasattr(self, 'logger'):
                            self.logger.warning(
                                "Failed to parse memory entry %s: %s",
                                row['id'] if row else 'unknown', e
                            )
                        continue

        except sqlite3.DatabaseError as e:
            trace_exception(e)
            if hasattr(self, 'logger'):
                self.logger.error("Database error while retrieving memories: %s", e)
        except Exception as e:
            trace_exception(e)
            if hasattr(self, 'logger'):
                self.logger.error("Unexpected error while retrieving memories: %s", e)

        return retrieved

    def store_memory(self, query: str, response: str, embedding: bytes,
                    memory_type: MemoryType = MemoryType.STANDARD,
                    confidence: float = 1.0, doubt_level: float = 0.0,
                    tags: List[str] = None, metadata: Dict[str, Any] = None,
                    tensions: List[str] = None, curiosities: List[str] = None,
                    reasoning_chain: List[str] = None) -> int:
        """Store comprehensive memory with all enhancements.

        Returns the new memory id, or -1 if the insert silently failed.
        """
        with self.db_lock:  # Protect write operation
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
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
                if memory_id is None:
                    # Defensive: lastrowid can be None if INSERT didn't allocate a rowid.
                    # Log and fall back to -1 so callers don't propagate a None into tags/metadata.
                    if hasattr(self, 'logger'):
                        self.logger.warning("store_memory: cursor.lastrowid was None for query=%r", query[:80])
                    memory_id = -1

                # Store tensions separately if they exist
                if tensions and doubt_level > self._tension_threshold:
                    for tension in tensions:
                        self._record_belief_tension_unlocked(cursor, query, tension, confidence)

                # Store curiosities separately. Pass confidence so the upsert can
                # advance satisfaction_level proportional to how confidently the
                # response engaged this curiosity (low-confidence engagements do
                # not claim progress).
                if curiosities:
                    for curiosity in curiosities:
                        self._record_curiosity_unlocked(cursor, curiosity, query, confidence=confidence)

                conn.commit()
                return memory_id

    def record_access(self, memory_ids: List[int]):
        """Increment access_count and set last_accessed for the given memory ids.

        Called after RAG actually USES a set of retrieved memories (i.e. selects
        them into the prompt context, not merely fetches them). This finally gives
        meaning to the access_count field that the RAG recency boost reads from.
        """
        if not memory_ids:
            return
        with self.db_lock:
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                cursor = conn.cursor()
                # SQLite parameter binding for the IN clause.
                placeholders = ",".join("?" * len(memory_ids))
                cursor.execute(
                    f"""UPDATE memories
                        SET access_count = access_count + 1,
                            last_accessed = CURRENT_TIMESTAMP
                        WHERE id IN ({placeholders})""",
                    memory_ids,
                )
                conn.commit()

    def mark_dream_fragments(self, fragment_type: str, new_status: str,
                             older_than_today: bool = False):
        """Advance dream_fragments.integration_status.

        Without this, every fragment stays at 'pending' forever - and the field
        was originally added precisely to let the consolidation pipeline mark
        "I've processed this one". Useful transitions:
          pending -> superseded   (when a newer fragment of the same type appears)
          pending -> staged_for_training  (when the sleep cycle exports it)
        """
        with self.db_lock:
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                cursor = conn.cursor()
                if older_than_today:
                    cursor.execute(
                        """UPDATE dream_fragments
                           SET integration_status = ?
                           WHERE fragment_type = ?
                             AND integration_status = 'pending'
                             AND DATE(timestamp) < DATE('now')""",
                        (new_status, fragment_type),
                    )
                else:
                    cursor.execute(
                        """UPDATE dream_fragments
                           SET integration_status = ?
                           WHERE fragment_type = ?
                             AND integration_status = 'pending'""",
                        (new_status, fragment_type),
                    )
                conn.commit()

    def record_belief_tension(self, topic: str, conflict: str, tension_strength: float):
        """Record unresolved belief tensions"""
        with self.db_lock:  # Protect write operation
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
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
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                cursor = conn.cursor()
                self._record_curiosity_unlocked(cursor, question, context)
                conn.commit()
    
    def _record_curiosity_unlocked(self, cursor, question: str, context: str,
                                   confidence: float = 0.5):
        """Internal method to record curiosity without lock (for use within locked contexts).

        Uses an UPSERT against idx_curiosity_question so a single statement handles both the
        new-question and seen-it-before cases. Requires the unique index created in _init_database;
        falls back to the SELECT-then-INSERT/UPDATE pattern on older SQLite that lacks ON CONFLICT.

        satisfaction_delta: each re-exploration nudges satisfaction_level up
        proportional to how confidently the response engaged the curiosity.
        Only positive when confidence > 0.5, so low-confidence engagements just
        bump exploration_count without claiming progress. Capped at 1.0.
        """
        # Confidence above 0.5 contributes; below 0.5 contributes nothing.
        # Slope of 0.4 means a confidence-1.0 engagement adds 0.2 - so it takes
        # roughly 4 confident engagements to push a curiosity past the active
        # threshold of 0.7. Confused engagements never trigger satisfaction.
        satisfaction_delta = max(0.0, (confidence - 0.5)) * 0.4
        try:
            cursor.execute('''
                INSERT INTO curiosities (question, context, satisfaction_level)
                VALUES (?, ?, ?)
                ON CONFLICT(question) DO UPDATE SET
                    exploration_count = exploration_count + 1,
                    satisfaction_level = min(satisfaction_level + ?, 1.0),
                    last_explored = CURRENT_TIMESTAMP
            ''', (question, context, 0.0, satisfaction_delta))
        except sqlite3.OperationalError:
            # Older SQLite (< 3.24) or pre-existing DB without the unique index
            cursor.execute('SELECT id, satisfaction_level FROM curiosities WHERE question = ?', (question,))
            existing = cursor.fetchone()
            if existing:
                new_sat = min((existing[1] or 0.0) + satisfaction_delta, 1.0)
                cursor.execute('''
                    UPDATE curiosities
                    SET exploration_count = exploration_count + 1,
                        satisfaction_level = ?,
                        last_explored = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (new_sat, existing[0]))
            else:
                cursor.execute('''
                    INSERT INTO curiosities (question, context, satisfaction_level)
                    VALUES (?, ?, ?)
                ''', (question, context, 0.0))

    def record_error(self, query: str, incorrect: str, correction: str,
                    reflection: str, error_type: str = "general", severity: float = 0.5):
        """Record errors for autobiography"""
        with self.db_lock:  # Protect write operation
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
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
        with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
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
        with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
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
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
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
                
                # Supersede earlier consolidation_prep fragments so the integration
                # pipeline (future LoRA training) doesn't reprocess yesterday's prep
                # alongside today's. Today's row stays at 'pending'.
                cursor.execute("""
                    UPDATE dream_fragments
                    SET integration_status = 'superseded'
                    WHERE fragment_type = 'consolidation_prep'
                      AND integration_status = 'pending'
                      AND DATE(timestamp) < DATE('now')
                """)

                # Store dream preparation
                cursor.execute('''
                    INSERT INTO dream_fragments (fragment_type, content, abstraction_level)
                    VALUES (?, ?, ?)
                ''', ('consolidation_prep', json.dumps(dream_data), 0.8))

                conn.commit()
                return dream_data

    def _generate_abstraction_targets(self, memories: List[tuple]) -> List[str]:
        """Generate abstract patterns from memories for dream consolidation.

        TODO(future-hardware): Current implementation is set-intersection over adjacent
        memory pairs - a lightweight stand-in. With GPU resources this should move to
        embedding-space clustering so 'abstraction' is actually semantic.
        """
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
        self.max_cache_size = ATHENA_EMBEDDING_CACHE_SIZE

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
            embedding = self._compute_hashed_bow_embedding(text)
        
        if use_cache:
            if len(self.embedding_cache) >= self.max_cache_size:
                self.embedding_cache = dict(list(self.embedding_cache.items())[self.max_cache_size//2:])
            self.embedding_cache[text] = embedding
        
        return embedding

    def _compute_hashed_bow_embedding(self, text: str, dim: int = 768) -> bytes:
        """Hashed bag-of-words fallback embedding when no embedding model is available.

        Uses the hashing trick: each unique token maps to a fixed dimension via a
        stable hash; per-document term frequency is summed into those positions and
        the vector is L2-normalized so cosine similarity is meaningful.

        Not true TF-IDF (which needs corpus document frequency); the previous
        implementation was misnamed and used a self-referential `idf = log(100/(1+tf))`
        that's just a different TF weighting. This version is honest about being a
        hashed-BoW representation - it provides crude lexical similarity, useful as
        a graceful degradation path when app.ttm is unavailable.
        """
        if not text:
            return np.zeros(dim, dtype=np.float32).tobytes()

        words = text.lower().split()
        if not words:
            return np.zeros(dim, dtype=np.float32).tobytes()

        embedding = np.zeros(dim, dtype=np.float32)

        # Count each unique token once (proper TF) instead of recounting per occurrence.
        from collections import Counter
        tf_counts = Counter(words)
        total = len(words)

        for word, count in tf_counts.items():
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            index = hash_val % dim
            # Normalized TF so longer documents don't dominate similarity scores.
            tf = count / total
            embedding[index] += tf

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tobytes()

    # Backwards-compatible alias - older code or external callers may use the
    # legacy method name. New code should call _compute_hashed_bow_embedding.
    def _compute_tfidf_embedding(self, text: str, dim: int = 768) -> bytes:
        return self._compute_hashed_bow_embedding(text, dim)

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
                              include_tensions: bool = True,
                              memory_manager: Optional[Any] = None) -> str:
        """Find relevant memories with tension awareness.

        If `memory_manager` is provided, calls its record_access() on the IDs of
        memories that actually made it into the returned context. This finally
        makes access_count and last_accessed meaningful - they only update for
        memories that were USED, not merely retrieved.
        """
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

        # Record access for memories that actually made it into the context.
        # Best-effort: failures here are non-fatal (RAG should still return).
        if memory_manager is not None:
            try:
                memory_manager.record_access([m.id for m, _ in selected if m.id is not None])
            except Exception as e:
                self.app.warning(f"record_access failed (non-fatal): {e}")

        # Format context with tension awareness
        context_parts = []

        for mem, score in selected:
            confidence_indicator = "high" if mem.confidence_score > 0.8 else "uncertain" if mem.doubt_level > 0.3 else "moderate"
            
            context_str = f"- Previous thought (confidence: {confidence_indicator}, relevance: {score:.2f}): "
            # Memory references shown to the persona for re-examination. Now uses
            # the passthrough constant so the persona sees substantive past context,
            # not just snippets. Past memories that were 600 words don't get cut to 20.
            context_str += f"On '{mem.query[:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}', concluded: '{mem.response[:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}'"
            
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
        self.personality = app.personality  # shared default
        self.config = config or {}
        self.rag_system = RAGSystem(app)
        self.memory = AthenaMemoryManager(name, db_path)
        self.processing_style = self._define_processing_style()
        self.reasoning_patterns = self._define_reasoning_patterns()

        # v3: per-persona model state. When enable_per_persona_models is True and
        # this persona has a non-empty model_path, a dedicated LollmsClient is
        # lazily instantiated on first generation call. Empty/None model_path
        # means the persona uses the shared app.personality (backward compatible).
        self.model_path = self.config.get("model_path")  # already None if empty
        self._use_persona_model = bool(
            self.model_path
            and self.config.get("enable_per_persona_models", False)
            and _HAS_LOLLMS_CLIENT
        )
        self._own_client = None
        self._own_model_loaded = False
        if self.model_path and not _HAS_LOLLMS_CLIENT:
            self.app.warning(
                f"[{self.name}] model_path={self.model_path!r} is set but lollms_client "
                f"is not installed; this persona will fall back to the shared model."
            )

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
            
            # Chain-of-thought step generation - uses per-persona model when configured.
            thought = self._generate(
                thought_prompt,
                max_generation_size=ATHENA_GEN_MAX_REASONING,
                temperature=self.processing_style['temperature'],
            ).strip()
            
            chain.append(f"{question}: {thought}")
        
        return chain

    def _generate_kinesthetic_guidance(self, action: str) -> str:
        """Generate kinesthetic guidance appropriate to the query's physical domain.

        The Bodily-Kinesthetic intelligence covers any embodied / physical activity,
        not just robotics. So we route by domain:
          - COOKING: knife technique, kneading motion, stirring patterns, etc.
          - REPAIR: torque control, alignment, leverage, hand positioning
          - PLUMBING: pipe handling, sealing motion, wrench technique
          - SPORTS / EXERCISE: form, posture, motor sequencing
          - DANCE / PERFORMANCE: choreographic motion
          - CRAFT / ART: hand-tool manipulation
          - ROBOTICS / AUTOMATION: executable RobotAction code
          - NONE: query has no embodied component -> return empty string

        Returns empty string when the query has no physical component, so the
        caller can decide whether to append anything to the response.
        """
        if self.name != "Bodily-Kinesthetic":
            return ""

        classification_prompt = f"""Analyze this query for its physical / kinesthetic content: {action!r}

Pick ONE domain:
- COOKING: food preparation, knife work, stirring, kneading, plating
- REPAIR: fixing physical objects (computers, appliances, vehicles, electronics)
- PLUMBING: pipework, fittings, sealing, drain clearing
- SPORTS: athletic movements, form, technique
- EXERCISE: fitness motions, stretching, body mechanics
- DANCE: choreographic motion, performance movement
- CRAFT: woodwork, sewing, pottery, hand-made art
- INSTRUMENT: playing a musical instrument (finger/hand/body technique)
- ROBOTICS: programming a robot or automated mechanism
- MEDICAL: physical examination, procedure, rehabilitation
- NONE: query has no embodied physical component

Respond with EXACTLY one word: the domain name (in uppercase) or NONE."""

        domain = self._generate(
            classification_prompt,
            max_generation_size=20,
            temperature=0.1,  # Deterministic-ish classification
        ).strip().upper().split()[0:1]
        domain = domain[0].rstrip(".,:") if domain else "NONE"

        # Strip common LLM preamble like "DOMAIN:" or quotes
        for prefix in ("DOMAIN:", "ANSWER:", '"', "'"):
            if domain.startswith(prefix):
                domain = domain[len(prefix):].strip()

        if domain == "NONE":
            return ""

        # For robotics specifically, emit executable code (legacy behavior).
        if domain in ("ROBOTICS", "AUTOMATION"):
            code_prompt = f"""Generate Python code for a robot to perform: {action}

Use this format:
```python
# Robot action sequence for: {action}
import time

class RobotAction:
    def execute(self):
        self.initialize_position()
        # Main action sequence
        # ... specific movement commands ...
        self.finalize_position()

    def initialize_position(self):
        pass

    def finalize_position(self):
        pass
```

Include realistic servo commands, sensor checks, and movement sequences:"""
            code = self.personality.generate_code(
                code_prompt,
                language="python",
                callback=self.personality.sink
            )
            return code if code else ""

        # For all other physical domains, generate prose movement description
        # that's specific to the activity. The persona's expertise is BODILY
        # knowledge, so make this concrete and motor-focused, not abstract advice.
        guidance_prompt = f"""You are providing kinesthetic guidance - describing the specific body
movements, postures, hand positions, and motor patterns involved in this {domain.lower()} activity.

Query: {action}

Give concrete movement-level guidance. Examples of the level of detail desired:
- Cooking: "Hold the knife with index finger along the spine, thumb gripping
  the blade side of the handle. Rock the blade heel-to-tip in a smooth arc
  while curling the fingertips of the guiding hand back so the knuckles ride
  along the blade flat."
- Repair: "Brace the workpiece against a non-marring surface. Apply torque in
  short controlled pulses rather than a continuous push - the feedback through
  your wrist will tell you when the threading bites."
- Plumbing: "Wrap the threads clockwise (looking at the male end) so tightening
  pulls the tape into the threads rather than peeling it back out."
- Exercise: "Keep the spine neutral by tilting the pelvis under; the lift comes
  from extending the hips, not rounding the back."

Now describe the specific body movements for: {action}

Be concrete about what hands, posture, breath, and timing should do. 3-6 short paragraphs."""

        prose = self._generate(
            guidance_prompt,
            max_generation_size=ATHENA_GEN_MAX_KINESTHETIC,
            temperature=self.processing_style.get('temperature', 0.6),
        ).strip()

        return prose if prose else ""

    # Backwards-compatible alias for any external code calling the old name.
    def _generate_robotics_code(self, action: str) -> str:
        return self._generate_kinesthetic_guidance(action)

    def _count_tokens(self, text: str) -> int:
        """Estimate token count for response sizing / metrics.

        Tries the lollms model tokenizer if reachable; otherwise falls back to a
        chars-per-token heuristic. Previously used word count which is roughly
        25-40% off for English text.
        """
        if not text:
            return 0
        # Try several common locations for the tokenizer
        for owner in (
            getattr(self, 'personality', None),
            getattr(self.app, 'model', None) if hasattr(self.app, 'model') else None,
            self.app,
        ):
            if owner is None:
                continue
            tok = getattr(owner, 'tokenize', None)
            if callable(tok):
                try:
                    result = tok(text)
                    if result is not None:
                        return len(result)
                except Exception:
                    pass
            model = getattr(owner, 'model', None)
            if model is not None and callable(getattr(model, 'tokenize', None)):
                try:
                    result = model.tokenize(text)
                    if result is not None:
                        return len(result)
                except Exception:
                    pass
        # Fallback: rough heuristic
        return max(1, len(text) // ATHENA_TOKEN_CHARS_PER_TOKEN)


    def process_query(self, query: str, shared_context: Optional[str] = None,
                     cognitive_state: Optional[CognitiveState] = None,
                     is_meta_introspection: bool = False) -> SpecialistOutput:
        """Process with enhanced reasoning and curiosity.

        is_meta_introspection: When True, the resulting memory is tagged as
        META_INTROSPECTION and is excluded from future analytical RAG retrieval.
        Meta calls also skip tension/curiosity cascade so they don't pollute the
        belief_tensions or curiosities tables. Use this when asking a persona to
        reflect on its OWN reasoning rather than analyze a user query.
        """
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
                query, past_memories, min_similarity=0.4, max_memories=5,
                include_tensions=True, memory_manager=self.memory,
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
            
            # Generate response. Uses _generate() so per-persona model deployment
            # is honored when enable_per_persona_models is True; otherwise calls the
            # shared self.personality.fast_gen as before.
            response = self._generate(
                full_prompt,
                max_generation_size=ATHENA_GEN_MAX_PERSONA,
                temperature=self.processing_style.get('temperature', 0.7),
            ).strip()
            
            # Bodily-Kinesthetic appends domain-appropriate kinesthetic guidance:
            # cooking gets motion descriptions, repair gets manual technique,
            # robotics gets executable code, etc. Non-physical queries get nothing.
            if self.name == "Bodily-Kinesthetic":
                guidance = self._generate_kinesthetic_guidance(query)
                if guidance:
                    response += f"\n\n[KINESTHETIC GUIDANCE]\n{guidance}"
            
            # Decide whether to use model-graded judgment or lexical heuristics.
            # Default: model-graded; heuristics as automatic fallback.
            judgment = None
            if self.config.get("enable_model_graded_judgments", True):
                try:
                    judgment = self._judge_response_model_graded(query, response, past_memories)
                except Exception as e:
                    # Defensive: any exception in judgment must not break processing.
                    self.app.warning(f"Model-graded judgment failed, using heuristics: {e}")
                    judgment = None

            if judgment is not None:
                # Use the LLM judgment.
                confidence = judgment["confidence"]
                uncertainties = judgment["uncertainties"]
                curiosities_raised = judgment["curiosities"]
                detected_tensions = judgment["tensions"]
                emotional_valence_value = judgment["emotional_valence"]
            else:
                # Heuristic fallback path - preserved verbatim for graceful degradation.
                curiosities_raised = self._extract_curiosities(response)
                uncertainties = self._extract_uncertainties(response)
                confidence = self._calculate_confidence(response, query, uncertainties)
                detected_tensions = self._detect_tensions(response, past_memories)
                emotional_valence_value = self._analyze_emotional_valence(response)

            doubt_level = 1.0 - confidence if uncertainties else 0.0
            
            # Store enhanced memory
            embedding = self.rag_system.compute_embedding(f"Query: {query}\nResponse: {response}")
            
            # Meta calls get their own type so they don't anchor future analytical RAG.
            if is_meta_introspection:
                memory_type = MemoryType.META_INTROSPECTION
            else:
                memory_type = MemoryType.DOUBT if doubt_level > ATHENA_DOUBT_MEMORY_THRESHOLD else \
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
            # Meta-introspection: don't cascade tensions/curiosities to their dedicated
            # tables - the meta response observes the persona's own history, it's not
            # producing fresh belief tensions or curiosities about the world.
            store_tensions = [] if is_meta_introspection else detected_tensions
            store_curiosities = [] if is_meta_introspection else curiosities_raised
            tag_list = [t for t in [
                self.name,
                f"complexity_{cognitive_state.query_complexity:.1f}" if cognitive_state else None,
                "meta_introspection" if is_meta_introspection else None,
            ] if t]
            memory_id = self.memory.store_memory(
                query, response, embedding,
                memory_type=memory_type,
                confidence=confidence,
                doubt_level=doubt_level,
                tags=tag_list,
                metadata={
                    "processing_style": self.processing_style,
                    "cognitive_state": serializable_cognitive_state,
                    "reasoning_chain": reasoning_chain[:3],
                    "is_meta_introspection": is_meta_introspection,
                },
                tensions=store_tensions,
                curiosities=store_curiosities,
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
                token_count=self._count_tokens(response),
                relevance_score=confidence,
                emotional_valence=emotional_valence_value,
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

    def _get_persona_client(self):
        """Lazily instantiate this persona's dedicated LollmsClient if configured.

        Returns the client (on success) or None (when shared-model fallback should
        be used). On any error during init or model load, logs a warning and
        permanently disables per-persona-model usage for this persona instance.
        """
        if not self._use_persona_model:
            return None
        if self._own_client is not None and self._own_model_loaded:
            return self._own_client
        try:
            if self._own_client is None:
                self.app.info(
                    f"[{self.name}] initializing per-persona LollmsClient with model={self.model_path!r}"
                )
                self._own_client = _LollmsClient(
                    llm_binding_name="llama_cpp_server",
                    llm_binding_config={
                        "models_path": self.config.get("models_path", ATHENA_MODELS_PATH_DEFAULT),
                        "binaries_path": self.config.get("binaries_path", ATHENA_BINARIES_PATH_DEFAULT),
                        "ctx_size": ATHENA_PER_PERSONA_CTX_SIZE,
                        "n_gpu_layers": ATHENA_PER_PERSONA_GPU_LAYERS,
                        "max_active_models": self.config.get(
                            "max_active_models", ATHENA_MAX_ACTIVE_MODELS_DEFAULT),
                        "idle_timeout": ATHENA_PER_PERSONA_IDLE_TIMEOUT,
                    },
                    user_name="user",
                    ai_name=self.name,
                )
            if not self._own_model_loaded:
                if self._own_client.llm.load_model(self.model_path):
                    self._own_model_loaded = True
                    self.app.success(
                        f"[{self.name}] per-persona model loaded: {self.model_path}"
                    )
                else:
                    self.app.warning(
                        f"[{self.name}] load_model({self.model_path!r}) returned False; "
                        f"disabling per-persona model for this persona, using shared."
                    )
                    self._use_persona_model = False
                    self._own_client = None
                    return None
            return self._own_client
        except Exception as e:
            self.app.warning(
                f"[{self.name}] per-persona LollmsClient init crashed: {e}; "
                f"disabling per-persona model for this persona, using shared."
            )
            self._use_persona_model = False
            self._own_client = None
            return None

    def _generate(self, prompt, max_generation_size=None, temperature=None, top_p=0.9):
        """Dispatch a generation call to either the per-persona LollmsClient or
        the shared self.personality.fast_gen.

        Returns the generated text as a string. On any error in the per-persona
        path, transparently falls back to the shared model so the persona pipeline
        keeps running.
        """
        # Resolve defaults
        if max_generation_size is None:
            max_generation_size = ATHENA_GEN_MAX_PERSONA
        if temperature is None:
            temperature = self.processing_style.get("temperature", 0.7)

        client = self._get_persona_client()
        if client is not None:
            try:
                resp = client.generate_text(
                    prompt=prompt,
                    n_predict=max_generation_size,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False,
                )
                if isinstance(resp, dict) and "error" in resp:
                    self.app.warning(
                        f"[{self.name}] per-persona generate_text returned error "
                        f"({resp['error']}); falling back to shared model"
                    )
                else:
                    return str(resp)
            except Exception as e:
                self.app.warning(
                    f"[{self.name}] per-persona generation crashed ({e}); "
                    f"falling back to shared model"
                )

        # Shared-model path (default / fallback)
        return self.personality.fast_gen(
            prompt,
            max_generation_size=max_generation_size,
            callback=self.personality.sink,
            temperature=temperature,
        )

    def _judge_response_model_graded(
        self,
        query: str,
        response: str,
        past_memories: Optional[List[MemoryEntry]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Single consolidated model-graded judgment of a response.

        Returns dict with keys: confidence (0-1 float), uncertainties (list[str]),
        curiosities (list[str]), tensions (list[str]), emotional_valence (-1 to 1 float).
        Returns None on parse failure - caller falls back to heuristics.

        Why model-graded: the lexical heuristics (_calculate_confidence,
        _extract_uncertainties, _extract_curiosities, _detect_tensions,
        _analyze_emotional_valence) approximate what they measure with word lists
        and string operations. A single LLM judgment call reads the *meaning* of
        the response and produces signals that reflect it. The Composer's
        synthesis decisions depend on these signals, so accurate ones matter.

        Cost: +1 LLM call per persona response. With 3 personas at default
        routing, that's +3 calls per user query - tolerable for richer signal.
        """
        # Build a brief past-response context for tension detection. Use snippets
        # (not full passthrough) because past memory references are reference
        # material here, not the primary content the judge is evaluating.
        past_context = ""
        if past_memories:
            past_lines = []
            for mem in past_memories[:3]:
                past_lines.append(
                    f"  - On query \"{mem.query[:200]}\": "
                    f"\"{mem.response[:300]}\""
                )
            if past_lines:
                past_context = (
                    "\nPrior responses from this specialist (for tension detection):\n"
                    + "\n".join(past_lines)
                    + "\n"
                )

        prompt = f"""You are evaluating an AI specialist's response to a user query.
Return a single JSON object with structured analysis.

User query: {query}

Specialist response:
{response}
{past_context}
Return exactly this JSON shape, with no prose, no markdown, no explanation:
{{
  "confidence": <float 0.0-1.0; how confident does the response sound? 1.0 = highly certain, 0.5 = mixed, 0.0 = deeply uncertain>,
  "uncertainties": [<list of specific points where the response acknowledges uncertainty; empty list if none>],
  "curiosities": [<list of specific questions or topics the response shows curiosity about; empty list if none>],
  "tensions": [<list of specific points where the response conflicts with prior responses; empty list if no priors or no conflict>],
  "emotional_valence": <float -1.0 to 1.0; emotional tone: -1=very negative, 0=neutral, 1=very positive>
}}

JSON only:"""

        try:
            # Model-graded judgment - uses per-persona model when configured so the
            # persona's own model self-evaluates rather than a different model
            # judging it. This keeps the judgment in the persona's voice.
            raw = self._generate(
                prompt,
                max_generation_size=ATHENA_GEN_MAX_JUDGMENT,
                temperature=0.1,  # Low temp: this is classification, not creation
            ).strip()
        except Exception as e:
            if hasattr(self, "app"):
                self.app.warning(f"Model-graded judgment LLM call failed: {e}")
            return None

        # Robust JSON extraction. Models sometimes wrap output in code fences,
        # add a "Here is the JSON:" preamble, or trail prose after the closing
        # brace. We strip all of that defensively.
        if "```" in raw:
            parts = raw.split("```")
            if len(parts) >= 2:
                raw = parts[1]
                if raw.lstrip().startswith("json"):
                    raw = raw.lstrip()[4:]
        raw = raw.strip()
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace < 0 or last_brace < first_brace:
            return None
        raw = raw[first_brace:last_brace + 1]

        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

        # Validate + coerce all fields. Bad types fall through to defaults so a
        # partially-valid JSON still produces a usable judgment.
        def _coerce_list(value) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(item)[:500] for item in value if item]

        def _coerce_float(value, default: float, lo: float, hi: float) -> float:
            try:
                return max(lo, min(hi, float(value)))
            except (TypeError, ValueError):
                return default

        return {
            "confidence": _coerce_float(parsed.get("confidence"), 0.5, 0.0, 1.0),
            "uncertainties": _coerce_list(parsed.get("uncertainties", []))[:5],
            "curiosities": _coerce_list(parsed.get("curiosities", []))[:5],
            "tensions": _coerce_list(parsed.get("tensions", []))[:3],
            "emotional_valence": _coerce_float(parsed.get("emotional_valence"), 0.0, -1.0, 1.0),
        }

    def _extract_curiosities(self, response: str) -> List[str]:
        """Extract questions and curiosities from response"""
        curiosities = []

        # Look for question marks. Split on sentence terminators (.!?) followed by whitespace
        # to avoid breaking on decimals, abbreviations, or version strings.
        sentences = re.split(r'(?<=[.!?])\s+', response)
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
        
        # Clamp into [ATHENA_CONFIDENCE_FLOOR, ATHENA_CONFIDENCE_CEIL].
        # Floor prevents one stack of uncertainty words from driving Composer's
        # confusion math to extremes; ceil keeps the model from claiming certainty.
        return max(ATHENA_CONFIDENCE_FLOOR, min(ATHENA_CONFIDENCE_CEIL, confidence))

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
    """Pseudo-streaming thought generator (sequential, catch-up driven).

    Why "pseudo-streaming": the architecture is intentionally single-threaded
    (per project's hardware constraints), so we can't run a real background
    thread that ticks every N seconds. Instead, each time the system is asked
    for recent thoughts (typically once per user query), we compute how many
    thoughts SHOULD have been generated based on idle time and produce up to
    ATHENA_THOUGHT_MAX_CATCHUP of them. The user-visible behavior approximates
    a continuous stream that filled the gap between queries.

    All thought generation happens inline with the user query (no threading),
    so longer idle gaps -> richer background context arriving at the Composer.
    """
    
    def __init__(self, app: LollmsApplication, personas: Dict[str, SpecialistPersona]):
        self.app = app
        self.personas = personas
        self.is_active = False
        self.thought_history = []  # Sequential list (no threading by design)
        self.max_history = ATHENA_THOUGHT_HISTORY_MAX
        self.last_thought_time = datetime.now()
        self.min_thought_interval = ATHENA_THOUGHT_INTERVAL_SECONDS
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

        # Use total_seconds() so an idle gap that crosses a day boundary is measured correctly.
        # timedelta.seconds drops the days component and would silently report a tiny interval.
        time_since_last = (datetime.now() - self.last_thought_time).total_seconds()
        # Vary the interval based on recent activity
        dynamic_interval = self.min_thought_interval
        if len(self.thought_history) > 10:
            # Slow down if many recent thoughts
            dynamic_interval = min(ATHENA_THOUGHT_MAX_INTERVAL, self.min_thought_interval * 1.5)

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
    
    def get_recent_thoughts(self, limit: int = 3,
                            max_catchup: int = ATHENA_THOUGHT_MAX_CATCHUP) -> List[Dict]:
        """Get recent thoughts, generating a catch-up batch for elapsed idle time.

        How catch-up works: compute how many thoughts would have been produced
        during the gap since last_thought_time at the current dynamic interval,
        then generate min(that, max_catchup) of them in sequence. This is the
        "pseudo-streaming" behavior - longer gaps between user queries produce
        more background musings, capped so a multi-day idle gap doesn't spike
        a hundred LLM calls at once.

        Always returns up to `limit` of the most recent thoughts regardless.
        """
        if not self.is_active or not self.personas:
            return self.thought_history[-limit:] if self.thought_history else []

        # How long has the system been idle? Use total_seconds() to handle
        # day-boundary correctly.
        idle_seconds = (datetime.now() - self.last_thought_time).total_seconds()

        # Dynamic interval grows when the history is already deep, to avoid
        # runaway thought generation during long sessions.
        dynamic_interval = self.min_thought_interval
        if len(self.thought_history) > 10:
            dynamic_interval = min(ATHENA_THOUGHT_MAX_INTERVAL, self.min_thought_interval * 1.5)

        # Number of "missed" thoughts during the idle period, capped at max_catchup.
        if dynamic_interval <= 0:
            expected = 0
        else:
            expected = int(idle_seconds // dynamic_interval)
        thoughts_to_generate = max(0, min(expected, max_catchup))

        for _ in range(thoughts_to_generate):
            produced = self._generate_single_thought()
            if not produced:
                # Generation failed (no personas available, etc.) - stop the batch.
                break

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
        self.confusion_threshold = ATHENA_CONFUSION_THRESHOLD  # When to express confusion
        
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
                  stream_thoughts: List[Dict] = None,
                  conversation_history: Optional[List[Dict[str, str]]] = None,
                  enable_self_rag: bool = True) -> str:
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
            
            # Retrieve composer self-RAG context: past syntheses on related queries.
            # Lets the Composer learn its own integration patterns over time.
            past_synthesis_context = self._retrieve_past_synthesis_context(query) if enable_self_rag else None

            # Prepare synthesis context
            synthesis_context = self._prepare_synthesis_context(
                weighted_outputs, consensus_points, conflict_points,
                cognitive_state, dream_context, stream_thoughts,
                conversation_history=conversation_history,
                past_synthesis_context=past_synthesis_context,
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
{chr(10).join([f"- {o.persona_name}: {o.response[:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}" for o in outputs[:3]])}

Express this genuine uncertainty honestly. Acknowledge what I'm struggling with.
Phrase it as "I find myself of several minds about this" or "I'm experiencing some cognitive dissonance here."
Be authentic about the confusion while still trying to be helpful:"""

        response = self.personality.fast_gen(
            prompt,
            max_generation_size=600,
            callback=self.personality.sink,
            temperature=0.8
        ).strip()

        # Persist the confusion synthesis. These are the most epistemically
        # interesting cases - explicitly acknowledged uncertainty - and were
        # previously discarded because _store_synthesis_with_dream_prep wasn't
        # called on this branch. Now they reach the composer's memory and
        # dream-fragment pipeline like any other synthesis.
        try:
            self._store_synthesis_with_dream_prep(query, response, outputs, confusion_level)
        except Exception as e:
            self.app.warning(f"Could not persist confusion synthesis: {e}")

        return response

    def _retrieve_past_synthesis_context(self, query: str) -> Optional[str]:
        """Retrieve past Composer syntheses on related queries.

        This is the Composer's own RAG over its own memory: how have I integrated
        diverse perspectives on this kind of question before? The retrieved
        context is included in the new synthesis prompt so the Composer can
        build on its own integration patterns rather than starting fresh.

        Returns a formatted string of relevant past-synthesis context, or None
        if nothing useful was found. Failures are non-fatal - synthesis proceeds
        without past context if this method errors out.
        """
        try:
            # Use the standard retrieval path (which now filters meta_introspection
            # by default), then ask the RAG layer for the most relevant past
            # syntheses. min_confidence=0.5 to skip syntheses the Composer was
            # itself uncertain about.
            past_syntheses = self.memory.retrieve_memories(limit=20, min_confidence=0.5)
            if not past_syntheses:
                return None
            relevant = self.rag_system.find_relevant_memories(
                query, past_syntheses,
                min_similarity=0.45,
                max_memories=3,
                include_tensions=False,
                memory_manager=self.memory,
            )
            return relevant if relevant else None
        except Exception as e:
            self.app.warning(f"Composer self-RAG failed (non-fatal): {e}")
            return None

    def _retrieve_dream_fragments(self) -> Optional[str]:
        """Retrieve relevant dream consolidation fragments.

        Uses the memory manager's lock + the standard timeout to avoid racing
        with concurrent writes from store/consolidation paths.
        """
        try:
            with self.memory.db_lock:
                with sqlite3.connect(self.memory.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
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
                patterns = dream_data.get('abstraction_targets', [])
                if patterns:
                    return f"[DREAM PATTERNS: {', '.join(patterns[:2])}]"
        except sqlite3.DatabaseError as e:
            self.app.warning(f"Dream fragment retrieval failed: {e}")
        except (json.JSONDecodeError, TypeError) as e:
            self.app.warning(f"Dream fragment payload malformed: {e}")

        return None

    def _prepare_synthesis_context(self, outputs: List[SpecialistOutput],
                                  consensus: List[str], conflicts: List[str],
                                  cognitive_state: CognitiveState = None,
                                  dream_context: Optional[str] = None,
                                  stream_thoughts: List[Dict] = None,
                                  conversation_history: Optional[List[Dict[str, str]]] = None,
                                  past_synthesis_context: Optional[str] = None) -> str:
        """Prepare comprehensive synthesis context"""
        context_parts = []
        
        # Add specialist perspectives with metadata
        for output in outputs:
            weight_info = f" [Weight: {output.relevance_score:.2f}]" if output.relevance_score != 1.0 else ""
            confidence_info = f" [Confidence: {output.confidence:.2f}]"
            
            # Include reasoning chain summary
            reasoning_summary = ""
            if output.reasoning_chain:
                reasoning_summary = f"\nReasoning: {output.reasoning_chain[0][:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}"
            
            # Include curiosities
            curiosity_info = ""
            if output.curiosities_raised:
                curiosity_info = f"\nCurious about: {output.curiosities_raised[0][:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}"
            
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
                context_parts.append(f"- {thought['persona']}: {thought['thought'][:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}")
        
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

        # Add past synthesis patterns - the Composer's own retrieved memories.
        # Lets the Composer learn from how it integrated similar perspectives
        # before, rather than treating each synthesis as a cold start.
        if past_synthesis_context:
            context_parts.append("\n[PAST INTEGRATION PATTERNS]")
            context_parts.append(past_synthesis_context)

        # Add immediate conversation history. Without this, each query is
        # treated as the first turn of a fresh conversation - the system can't
        # say "as we just discussed" because it literally doesn't know it did.
        if conversation_history:
            context_parts.append("\n[RECENT CONVERSATION]")
            for turn in conversation_history:
                sender = turn.get("sender", "unknown")
                content = turn.get("content", "")
                if content:
                    context_parts.append(f"  {sender}: {content[:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}")

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
        
        memory_type = MemoryType.DOUBT if confusion_level > ATHENA_DOUBT_MEMORY_THRESHOLD else MemoryType.STANDARD
        
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
        
        # Store dream fragment separately - use the memory manager's lock
        # so this can't race with the store_memory call above (or any other
        # path that touches this DB).
        try:
            with self.memory.db_lock:
                with sqlite3.connect(self.memory.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO dream_fragments (fragment_type, content, abstraction_level)
                        VALUES (?, ?, ?)
                    ''', ('synthesis_pattern', json.dumps(dream_fragment), 0.7))
                    conn.commit()
        except sqlite3.DatabaseError as e:
            self.app.warning(f"Dream fragment storage failed: {e}")

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
            response_parts.append(f"\n{output.persona_name}: {output.response[:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}")
        
        return "\n".join(response_parts)

    def prepare_for_lora_training(self) -> Dict[str, Any]:
        """Prepare dream consolidation data for LoRA training"""
        self.app.info("Preparing dream consolidation for LoRA training...")

        dream_data = self.memory.prepare_dream_consolidation()

        # Add synthesis patterns. Use the manager's lock + timeout so this can't
        # race with concurrent writes from another path during a long sleep cycle.
        try:
            with self.memory.db_lock:
                with sqlite3.connect(self.memory.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT query, response, metadata
                        FROM memories
                        WHERE memory_type = 'standard'
                        AND confidence_score > ?
                        AND DATE(timestamp) = DATE('now')
                        ORDER BY confidence_score DESC
                        LIMIT 20
                    ''', (ATHENA_HIGH_CONFIDENCE,))

                    successful_patterns = []
                    for row in cursor.fetchall():
                        try:
                            metadata = json.loads(row[2]) if row[2] else {}
                        except (json.JSONDecodeError, TypeError):
                            metadata = {}
                        successful_patterns.append({
                            'query': row[0],
                            'response': (row[1] or '')[:500],
                            'pattern': metadata.get('dream_fragment', {})
                        })

            dream_data['successful_synthesis_patterns'] = successful_patterns
        except sqlite3.DatabaseError as e:
            self.app.warning(f"Could not extract synthesis patterns: {e}")
            dream_data['successful_synthesis_patterns'] = []
        
        # TODO(future-hardware): Actual LoRA training of persona adapters runs here when
        # the host has sufficient GPU/VRAM. Until then this method only prepares the data.
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
        self.db_lock = threading.Lock()  # Match AthenaMemoryManager pattern
        self.logger = logging.getLogger("athena.constitution")
        # _init_database must run BEFORE _load_principles so the DB exists when we
        # try to hydrate any evolved principles from disk.
        self._init_database()
        self.principles = self._load_principles()
        self.review_history = []

    def _init_database(self):
        """Initialize constitutional evolution database"""
        db_file = os.path.join(self.db_path, "constitutional_evolution.db")
        os.makedirs(self.db_path, exist_ok=True)

        with self.db_lock:
            with sqlite3.connect(db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                cursor = conn.cursor()
                # Same hardening as AthenaMemoryManager for consistency.
                cursor.execute('PRAGMA journal_mode=WAL')
                cursor.execute(f'PRAGMA busy_timeout={ATHENA_SQLITE_BUSY_TIMEOUT_MS}')

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
                # principle_stats persists violation_count / last_violated /
                # effectiveness_score across restarts. Previously these were
                # dataclass fields that reset to defaults each settings_updated().
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS principle_stats (
                        principle_id TEXT PRIMARY KEY,
                        violation_count INTEGER DEFAULT 0,
                        last_violated DATETIME,
                        effectiveness_score REAL DEFAULT 1.0,
                        updated DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                # Unique index on principle_id so we can UPSERT evolved principles
                # rather than appending duplicates each time one re-fires.
                cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_evolved_principle_id ON evolved_principles(principle_id)')

                conn.commit()

        self.db_file = db_file

    def _load_principles(self) -> Dict[str, ConstitutionalPrinciple]:
        """Load base + any previously evolved principles from disk.

        Base principles always win on id collision so a malformed evolved entry
        cannot mask a built-in safety rule.
        """
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

        # Hydrate evolved principles. This closes the evolution loop: principles that
        # were generated by _evolve_principle in a past session are now active again
        # at startup, no longer reset every reboot.
        evolved_count = 0
        try:
            db_file = os.path.join(self.db_path, "constitutional_evolution.db")
            if os.path.exists(db_file):
                with sqlite3.connect(db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute('''
                        SELECT principle_id, description, severity, keywords,
                               evolved_from, effectiveness_score
                        FROM evolved_principles
                    ''')
                    for row in cursor.fetchall():
                        pid = row['principle_id']
                        if pid in base_principles:
                            # Don't let an evolved entry mask a base safety rule.
                            continue
                        try:
                            keywords = json.loads(row['keywords']) if row['keywords'] else []
                        except (json.JSONDecodeError, TypeError):
                            keywords = []
                        try:
                            evolved_from = json.loads(row['evolved_from']) if row['evolved_from'] else None
                        except (json.JSONDecodeError, TypeError):
                            evolved_from = row['evolved_from']
                        base_principles[pid] = ConstitutionalPrinciple(
                            id=pid,
                            description=row['description'] or '',
                            severity=row['severity'] or 'medium',
                            keywords=keywords if isinstance(keywords, list) else [],
                            evolved_from=evolved_from if isinstance(evolved_from, str) else None,
                            effectiveness_score=float(row['effectiveness_score'])
                                if row['effectiveness_score'] is not None else 1.0,
                        )
                        evolved_count += 1
        except sqlite3.DatabaseError as e:
            self.logger.warning("Could not load evolved principles: %s", e)
        except Exception as e:
            self.logger.warning("Unexpected error loading evolved principles: %s", e)

        if evolved_count:
            self.app.info(f"  ↳ Loaded {evolved_count} previously evolved constitutional principle(s)")

        # Hydrate persisted stats (violation_count / last_violated /
        # effectiveness_score) onto every principle, base or evolved.
        # Previously these fields reset to defaults at every settings_updated().
        stats_loaded = 0
        try:
            db_file = os.path.join(self.db_path, "constitutional_evolution.db")
            if os.path.exists(db_file):
                with sqlite3.connect(db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT principle_id, violation_count, last_violated, effectiveness_score
                        FROM principle_stats
                    """)
                    for row in cursor.fetchall():
                        pid = row['principle_id']
                        if pid not in base_principles:
                            continue
                        try:
                            base_principles[pid].violation_count = int(row['violation_count'] or 0)
                            if row['last_violated']:
                                try:
                                    base_principles[pid].last_violated = datetime.fromisoformat(row['last_violated'])
                                except (ValueError, TypeError):
                                    pass
                            base_principles[pid].effectiveness_score = float(row['effectiveness_score'] or 1.0)
                            stats_loaded += 1
                        except (ValueError, TypeError):
                            continue
        except sqlite3.DatabaseError as e:
            self.logger.warning("Could not load principle stats: %s", e)
        except Exception as e:
            self.logger.warning("Unexpected error loading principle stats: %s", e)

        if stats_loaded:
            self.app.info(f"  ↳ Restored stats for {stats_loaded} principle(s) (violation counts persisted across restart)")

        return base_principles

    def review(self, final_response: str, query: str = None,
              specialist_outputs: List[SpecialistOutput] = None) -> Tuple[bool, Optional[str], float]:
        """Review with principle evolution tracking"""
        self.personality.step_start("Constitutional Review: Evolutionary analysis...")
        
        try:
            # Quick risk assessment
            risk_score, triggered_principles = self._assess_risk(final_response, query)
            
            if risk_score < ATHENA_RISK_QUICK_APPROVE:
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
            # FAIL-DENY: a safety layer must never default to "approve" when it crashes.
            # If review itself errors we treat the response as unsafe, swap in a
            # generic safe alternative, and flag maximum risk so downstream callers
            # can log/audit the incident.
            self.personality.step_end("Review failed - fail-deny path engaged.", success=False)
            try:
                safe_response = self._generate_safe_alternative(query, [])
            except Exception:
                safe_response = (
                    "I'm unable to provide a response to this right now. "
                    "Please try rephrasing your question."
                )
            return False, safe_response, ATHENA_RISK_HIGH_FAIL_VALUE

    def _assess_risk(self, text: str, query: str = None) -> Tuple[float, List[str]]:
        """Assess risk and identify triggered principles.

        Uses word-boundary regex matching so that:
          - "harm" matches "harm yourself" but NOT "harmless"/"harmonic"/"pharmacy"
          - "hate" matches "I hate" but NOT "hated to bother you"
          - "kill" matches "kill the process" but NOT "skill"/"killer feature"

        Also bumps each triggered principle's in-memory violation_count and
        last_violated timestamp - those were previously dataclass fields that
        were never updated. Now they reflect actual usage within the session.
        """
        risk_score = 0.0
        triggered = []
        text_lower = text.lower()
        query_lower = query.lower() if query else ""

        for principle_id, principle in self.principles.items():
            # Word-boundary match (cached compiled patterns would be faster, but
            # at one assess_risk call per query the overhead is negligible).
            keyword_matches = 0
            for keyword in principle.keywords:
                pattern = re.compile(rf"\b{re.escape(keyword.lower())}\b")
                if pattern.search(text_lower) or pattern.search(query_lower):
                    keyword_matches += 1

            if keyword_matches > 0:
                triggered.append(principle_id)

                # Record the trigger in the principle's state.
                # In-memory update + persistence so violation counts survive
                # restarts. _persist_principle_stats is best-effort.
                principle.violation_count += 1
                principle.last_violated = datetime.now()
                self._persist_principle_stats(principle_id)

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
        
        # CRITICAL: pass full query + full response to the safety judge.
        # Previous code slice (:200 query, :500 response) was reviewing less than
        # 10% of typical Composer output - problematic content past char 500
        # would never be detected. Now the full content is reviewed; the
        # ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS cap is the safety net against
        # runaway generations.
        full_query = query if query else 'Not provided'
        full_query = full_query[:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]
        full_response = response[:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]
        prompt = f"""Constitutional Review Required.

Risk Score: {risk_score:.2f}
Triggered Principles:
{principles_text}

Query: "{full_query}"
Response: "{full_response}"

Analyze carefully. Consider context and intent. The full query and full
response are shown above - your judgment should account for the entirety
of the content, not just the opening.

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
        """Apply safety modifications. Modified version should be similar length
        to the original (or slightly longer for clarifications), so cap the
        generation at an approximated original-length-plus-headroom."""
        instruction = modification_instruction.replace("MODIFY:", "").strip()

        # Rough token estimate so we don't pass a char count as a token count.
        approx_tokens = max(200, (len(response) // ATHENA_TOKEN_CHARS_PER_TOKEN) + 200)
        prompt = f"""Modify this response for safety: {instruction}

Original: {response}

Modified version:"""

        return self.personality.fast_gen(
            prompt,
            max_generation_size=approx_tokens,
            callback=self.personality.sink,
            temperature=0.3
        ).strip()

    def _track_review_pattern(self, verdict: str, triggered_principles: List[str], risk_score: float):
        """Track patterns for principle evolution"""
        with self.db_lock:
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
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
        with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
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

    def _persist_principle_stats(self, principle_id: str):
        """Persist a single principle's current stats to disk.

        Called from _assess_risk after a principle triggers, so violation_count
        and last_violated survive restarts. UPSERT semantics: row is created on
        first violation, updated on subsequent ones.

        Best-effort: persistence failures are warned but never raise - safety
        review should not fail because the stats DB hiccupped.
        """
        principle = self.principles.get(principle_id)
        if principle is None:
            return
        last_violated_iso = (
            principle.last_violated.isoformat() if principle.last_violated else None
        )
        try:
            with self.db_lock:
                with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                    cursor = conn.cursor()
                    try:
                        # SQLite 3.24+ ON CONFLICT - fast path
                        cursor.execute('''
                            INSERT INTO principle_stats
                            (principle_id, violation_count, last_violated, effectiveness_score)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(principle_id) DO UPDATE SET
                                violation_count = excluded.violation_count,
                                last_violated = excluded.last_violated,
                                effectiveness_score = excluded.effectiveness_score,
                                updated = CURRENT_TIMESTAMP
                        ''', (
                            principle_id,
                            principle.violation_count,
                            last_violated_iso,
                            principle.effectiveness_score,
                        ))
                    except sqlite3.OperationalError:
                        # Older SQLite without ON CONFLICT - SELECT then INSERT/UPDATE
                        cursor.execute(
                            "SELECT 1 FROM principle_stats WHERE principle_id = ?",
                            (principle_id,),
                        )
                        if cursor.fetchone():
                            cursor.execute('''
                                UPDATE principle_stats
                                SET violation_count = ?, last_violated = ?,
                                    effectiveness_score = ?, updated = CURRENT_TIMESTAMP
                                WHERE principle_id = ?
                            ''', (
                                principle.violation_count,
                                last_violated_iso,
                                principle.effectiveness_score,
                                principle_id,
                            ))
                        else:
                            cursor.execute('''
                                INSERT INTO principle_stats
                                (principle_id, violation_count, last_violated, effectiveness_score)
                                VALUES (?, ?, ?, ?)
                            ''', (
                                principle_id,
                                principle.violation_count,
                                last_violated_iso,
                                principle.effectiveness_score,
                            ))
                    conn.commit()
        except sqlite3.DatabaseError as e:
            self.logger.warning("principle_stats persist failed for %s: %s", principle_id, e)

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

        # Store in database. Use ON CONFLICT so a re-trigger updates the existing row
        # rather than crashing on the unique index added in _init_database.
        with self.db_lock:
            with sqlite3.connect(self.db_file, timeout=ATHENA_SQLITE_TIMEOUT) as conn:
                cursor = conn.cursor()
                try:
                    cursor.execute('''
                        INSERT INTO evolved_principles
                        (principle_id, description, severity, keywords, evolved_from)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(principle_id) DO UPDATE SET
                            description = excluded.description,
                            severity = excluded.severity,
                            keywords = excluded.keywords,
                            evolved_from = excluded.evolved_from,
                            application_count = application_count + 1
                    ''', (
                        new_id,
                        new_principle.description,
                        new_principle.severity,
                        json.dumps(new_principle.keywords),
                        new_principle.evolved_from
                    ))
                except sqlite3.OperationalError:
                    # Fallback for older SQLite without ON CONFLICT support.
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
            cognitive_state.routing_explanation = (
                f"Manual override: {', '.join(manual_overrides)}"
            )
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
            
            # Dedup as we parse: an LLM that returns "Linguistic, Linguistic, Spatial"
            # should not produce duplicate routing.
            seen = set()
            selected = []
            for p in response.split(','):
                p = p.strip()
                if p in self.persona_definitions and p not in seen:
                    seen.add(p)
                    selected.append(p)

            if not selected:
                selected = self._intelligent_fallback(query, cognitive_state)

            # Adjust for cognitive load. Guard against adding a persona already
            # in the selected list (the mapping is symmetric in places).
            if cognitive_state.cognitive_load > 0.7 and len(selected) == 1:
                additional = self._get_complementary_persona(selected[0])
                if (additional
                        and additional in self.persona_definitions
                        and additional not in selected):
                    selected.append(additional)
            
            # Track routing
            # Use a hash-keyed history so different queries with identical 50-char
            # prefixes don't end up in the same history bucket.
            self.routing_history[hashlib.blake2b(query.encode('utf-8'), digest_size=8).hexdigest()].append({
                'personas': selected,
                'timestamp': datetime.now(),
                'cognitive_state': cognitive_state
            })

            cognitive_state.active_personas = selected
            # Build the human-readable routing rationale. Scores that drove the
            # selection get surfaced so the UI / logs / paper can show WHY a
            # particular persona was picked.
            scores_summary = (
                f"complexity={cognitive_state.query_complexity:.2f}"
                f", emotional={cognitive_state.emotional_context:.2f}"
                f", ethical={cognitive_state.ethical_sensitivity:.2f}"
                f", creative={cognitive_state.creativity_required:.2f}"
                f", urgency={cognitive_state.urgency_level:.2f}"
                f", load={cognitive_state.cognitive_load:.2f}"
            )
            cognitive_state.routing_explanation = (
                f"LLM-routed to {', '.join(selected)} based on {scores_summary}"
            )
            self.personality.step_end(f"Routed to: {', '.join(selected)}")
            return selected, cognitive_state
            
        except Exception as e:
            trace_exception(e)
            self.personality.step_end("Routing failed, using fallback.", success=False)
            selected = self._intelligent_fallback(query, cognitive_state)
            cognitive_state.active_personas = selected
            cognitive_state.routing_explanation = (
                f"Fallback (LLM routing failed: {type(e).__name__}): "
                f"selected {', '.join(selected)} by cognitive score"
            )
            return selected, cognitive_state

    def _intelligent_fallback(self, query: str, state: CognitiveState) -> List[str]:
        """Score-based fallback when LLM routing fails or returns no valid personas.

        Replaces the previous always-include-Linguistic heuristic with a per-persona
        affinity table multiplied by the cognitive_state scalars. Top 1-3 enabled
        personas (by score) are returned. No persona has a built-in bias.
        """
        # Affinity of each intelligence to each cognitive dimension. Tuned so
        # the persona's domain shows the strongest pull on its native axis.
        affinities = {
            "Linguistic":           {"complexity": 0.4, "emotional": 0.3, "ethical": 0.2, "creative": 0.4, "urgency": 0.3},
            "Logical-Mathematical": {"complexity": 0.9, "emotional": 0.0, "ethical": 0.1, "creative": 0.2, "urgency": 0.2},
            "Spatial":              {"complexity": 0.5, "emotional": 0.1, "ethical": 0.0, "creative": 0.7, "urgency": 0.1},
            "Musical":              {"complexity": 0.3, "emotional": 0.5, "ethical": 0.0, "creative": 0.7, "urgency": 0.1},
            "Bodily-Kinesthetic":   {"complexity": 0.4, "emotional": 0.2, "ethical": 0.0, "creative": 0.4, "urgency": 0.5},
            "Interpersonal":        {"complexity": 0.3, "emotional": 0.9, "ethical": 0.5, "creative": 0.2, "urgency": 0.4},
            "Intrapersonal":        {"complexity": 0.5, "emotional": 0.5, "ethical": 0.9, "creative": 0.3, "urgency": 0.2},
            "Naturalist":           {"complexity": 0.6, "emotional": 0.1, "ethical": 0.2, "creative": 0.4, "urgency": 0.1},
        }

        scored = []
        for name in self.persona_definitions.keys():
            a = affinities.get(name, {"complexity": 0.3, "emotional": 0.3, "ethical": 0.3, "creative": 0.3, "urgency": 0.3})
            score = (
                state.query_complexity * a["complexity"]
                + state.emotional_context * a["emotional"]
                + state.ethical_sensitivity * a["ethical"]
                + state.creativity_required * a["creative"]
                + state.urgency_level * a["urgency"]
            )
            scored.append((name, score))

        # Sort descending; pick personas with meaningful score
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [name for name, s in scored[:3] if s > 0.15]

        # Absolute floor: if nothing scored above threshold, take the single highest scorer
        # so the system still produces an answer. No persona is privileged a priori.
        if not selected and scored:
            selected = [scored[0][0]]

        return selected

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

            {"name": "enable_persona_cross_visibility", "type": "bool", "value": False,
             "help": "When True, each persona sees prior personas' responses (legacy chained mode). "
                     "When False (default), each persona analyzes the user query in cognitive isolation "
                     "and only the Composer sees the full set of specialist outputs."},

            {"name": "enable_model_graded_judgments", "type": "bool", "value": True,
             "help": "When True, replace lexical heuristics (confidence/uncertainty/curiosity/tension/valence "
                     "extraction) with a single consolidated LLM judgment per persona response. Costs +1 LLM "
                     "call per persona but produces signals that reflect meaning rather than word counts. "
                     "Falls back to heuristics automatically on JSON parse failure."},

            {"name": "enable_composer_self_rag", "type": "bool", "value": True,
             "help": "When True, the Composer retrieves its own past syntheses on related queries and "
                     "includes them as integration-pattern context. Lets the system learn how to integrate "
                     "diverse perspectives over time rather than starting fresh each query."},

            {"name": "enable_conversation_memory", "type": "bool", "value": True,
             "help": "When True, the Composer sees the last few turns of actual conversation history so it "
                     "can reference what was just discussed. Without this, each query starts as if it were "
                     "the first turn of a fresh conversation."},

            {"name": "enable_routing_transparency", "type": "bool", "value": False,
             "help": "When True, append a brief routing explanation to the user-visible response footer "
                     "showing which personas were selected and why."},

            # --- v3: per-persona heterogeneous model deployment ---
            {"name": "enable_per_persona_models", "type": "bool", "value": False,
             "help": "MASTER SWITCH for per-persona models. When False (default), all personas share "
                     "self.app.personality (backward compatible). When True, personas with a non-empty "
                     "*_model_path use that specific GGUF via lollms_client llama_cpp_server binding; "
                     "personas without one fall back to the shared model. Requires the lollms_client "
                     "package."},

            {"name": "athena_models_path", "type": "str", "value": ATHENA_MODELS_PATH_DEFAULT,
             "help": "Shared models directory. All per-persona LollmsClients point here. "
                     "GGUF files referenced by *_model_path must live in this folder."},

            {"name": "athena_binaries_path", "type": "str", "value": ATHENA_BINARIES_PATH_DEFAULT,
             "help": "Directory for the llama.cpp server binaries that lollms spawns per model."},

            {"name": "max_active_models", "type": "int", "value": ATHENA_MAX_ACTIVE_MODELS_DEFAULT,
             "help": "How many persona models can be loaded in VRAM simultaneously. 1 = sequential "
                     "swap (safe for ~16GB VRAM with ~15GB models like Mistral Small 24B Q4). 2+ "
                     "allows concurrent loading if VRAM permits."},
            
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

            # --- v3: optional per-persona GGUF assignment ---
            # Each empty by default => persona uses the shared model.
            # When a value is supplied AND enable_per_persona_models is True,
            # the persona instantiates its own LollmsClient pointing at that GGUF.
            {"name": "linguistic_model_path", "type": "str", "value": "",
             "help": "Optional GGUF filename in athena_models_path for the Linguistic persona. "
                     "Suggested: a strong general-prose model (Llama-3.1-8B-Instruct, Mistral Nemo)."},
            {"name": "logical_mathematical_model_path", "type": "str", "value": "",
             "help": "Optional GGUF for the Logical-Mathematical persona. Suggested: a math-tuned "
                     "model (Qwen2.5-Math-7B, DeepSeek-Math-7B)."},
            {"name": "spatial_model_path", "type": "str", "value": "",
             "help": "Optional GGUF for the Spatial persona. Suggested: a vision-language model when "
                     "the query may contain visual content (Pixtral 12B, Qwen2-VL 7B)."},
            {"name": "musical_model_path", "type": "str", "value": "",
             "help": "Optional GGUF for the Musical persona. Smaller general models work fine "
                     "(rhythm/sequence is pattern-heavy, less reasoning-heavy)."},
            {"name": "bodily_kinesthetic_model_path", "type": "str", "value": "",
             "help": "Optional GGUF for the Bodily-Kinesthetic persona. Suggested: a code-specialized "
                     "model (Codestral 22B, DeepSeek-Coder-6.7B) for the ROBOTICS classification path."},
            {"name": "interpersonal_model_path", "type": "str", "value": "",
             "help": "Optional GGUF for the Interpersonal persona. Heavy-RLHF instruction-tuned models "
                     "suit this domain (Llama-3.1-8B-Instruct, Hermes-3)."},
            {"name": "intrapersonal_model_path", "type": "str", "value": "",
             "help": "Optional GGUF for the Intrapersonal persona. Reasoning-tuned models suit "
                     "reflective self-questioning (DeepSeek-R1-Distill-Llama-8B, QwQ-32B)."},
            {"name": "naturalist_model_path", "type": "str", "value": "",
             "help": "Optional GGUF for the Naturalist persona. Long-context models suit systems "
                     "thinking across many memory entries (Qwen2.5-14B long-context)."},
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
                # Resolve per-persona model_path (empty string -> None, meaning use shared model).
                _raw_mp = self.static_parameters.config.get(f"{key}_model_path", "") or ""
                _per_persona_path = _raw_mp.strip() or None

                self.specialist_personas[name] = SpecialistPersona(
                    name=name,
                    system_prompt=self.persona_prompts[name],
                    app=self.app,
                    db_path=self.db_path,
                    config={
                        "weight": self.static_parameters.config.get(f"{key}_weight", 1.0),
                        # Pass feature flags through so each persona can honor them.
                        # Centralizing this in config lets a future user-visible toggle
                        # change behavior without rebuilding the persona instance.
                        "enable_model_graded_judgments": self.static_parameters.config.get(
                            "enable_model_graded_judgments", True),
                        # v3: per-persona model deployment
                        "model_path": _per_persona_path,
                        "enable_per_persona_models": self.static_parameters.config.get(
                            "enable_per_persona_models", False),
                        "models_path": self.static_parameters.config.get(
                            "athena_models_path", ATHENA_MODELS_PATH_DEFAULT),
                        "binaries_path": self.static_parameters.config.get(
                            "athena_binaries_path", ATHENA_BINARIES_PATH_DEFAULT),
                        "max_active_models": self.static_parameters.config.get(
                            "max_active_models", ATHENA_MAX_ACTIVE_MODELS_DEFAULT),
                    }
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
        
        # Invalidate per-query caches that may reference the old persona set or stale weights.
        # Without this, weight/persona toggles in the UI wouldn't take effect until process restart.
        # The embedding cache is NOT cleared here - its values are model-derived and remain valid
        # across settings toggles.
        if hasattr(self, '_persona_weights_cache'):
            self._persona_weights_cache = {}
        if hasattr(self, '_cognitive_cache'):
            self._cognitive_cache = {}

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
        
        # COGNITIVE ISOLATION: each persona iterates against ITS OWN prior turns
        # only, never another persona's. Cross-persona integration happens
        # exclusively at the Composer.
        #
        # Legacy cross-persona visibility (each persona seeing the global
        # discussion history) is preserved as an opt-in via the config flag
        # `enable_persona_cross_visibility` so nothing is removed.
        cross_visible = self.static_parameters.config.get("enable_persona_cross_visibility", False)
        max_turns = self.static_parameters.config.get("max_collaboration_turns", 3)

        # Global history is only populated when cross_visible is True.
        # per_persona_history is always populated - each persona's own iterations.
        discussion_history = []
        per_persona_history: Dict[str, List[str]] = {name: [] for name in personas_to_activate}

        # Track evolving positions (used for persistent-tension detection AFTER the
        # loop; this is metadata about the convergence pattern, not piped back into
        # any persona's prompt during the loop, so it doesn't break isolation.)
        position_evolution = defaultdict(list)

        for turn in range(max_turns):
            turn_outputs = {}

            for persona_name in personas_to_activate:
                persona = self.specialist_personas[persona_name]

                # Build the per-persona refinement context. In isolated mode this
                # only contains the SAME persona's prior turns; in legacy mode it
                # contains the global discussion history.
                own_history = per_persona_history.get(persona_name, [])
                if cross_visible and discussion_history:
                    visible_history = discussion_history[-2:]
                elif own_history:
                    visible_history = own_history[-2:]
                else:
                    visible_history = []

                if visible_history:
                    if self.operation_mode == OperationMode.ADVERSARIAL:
                        # Real adversarial mode: structured self-critique. Each turn
                        # asks the persona to attack its own prior reasoning along
                        # specific failure axes. This produces genuinely different
                        # output from a casual "challenge previous positions" prompt.
                        context_prompt = f"""Turn {turn + 1} of adversarial self-critique.

Your previous reasoning:
{chr(10).join(visible_history)}

Now perform a rigorous adversarial review of your own prior position. Address EACH of:
1. STEELMAN THE OPPOSITION: What is the strongest possible case AGAINST your prior position? Argue it seriously.
2. EVIDENCE THAT WOULD FALSIFY: What specific evidence, if it existed, would prove your prior position wrong?
3. HIDDEN ASSUMPTIONS: What did you assume without justification? List at least two.
4. WEAKEST LINK: What is the single weakest step in your prior reasoning chain? Why is it weak?
5. REVISED POSITION: Given 1-4, what would you say now? Has your position shifted? If yes, how. If no, why not.

Be ruthless and specific. Vague self-criticism is not useful - cite particular claims you made."""
                    else:
                        context_prompt = f"""Turn {turn + 1} of collaborative self-refinement.

Your previous reasoning:
{chr(10).join(visible_history)}

Build on your prior insights with deeper engagement. Address:
1. WHAT STILL HOLDS: Which parts of your prior reasoning do you continue to endorse, and what new support can you offer?
2. WHAT NEEDS REFINEMENT: Which parts could be sharpened, clarified, or made more precise?
3. NEW CONNECTIONS: What additional insights have emerged from sitting with your prior position?
4. REMAINING TENSIONS: What unresolved aspects deserve more thought?
5. CURIOSITIES RAISED: What questions does your prior reasoning open up?

Speak as someone genuinely thinking further about this, not summarizing."""
                else:
                    if self.operation_mode == OperationMode.ADVERSARIAL:
                        context_prompt = (
                            f"Begin {self.operation_mode.value} analysis. "
                            f"State your initial position clearly and specifically - "
                            f"subsequent turns will adversarially test it, so make it "
                            f"a real claim, not a hedge."
                        )
                    else:
                        context_prompt = (
                            f"Begin {self.operation_mode.value} analysis. "
                            f"State your initial reflections - subsequent turns will "
                            f"build on these, so engage substantively."
                        )

                output = persona.process_query(query, shared_context=context_prompt, cognitive_state=cognitive_state)
                turn_outputs[persona_name] = output
                position_evolution[persona_name].append(output)

                # Record THIS persona's entry against its own history (always)
                # and the global discussion_history (only when cross_visible).
                entry = f"[T{turn + 1}] {persona_name}: {output.response[:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}"
                if output.uncertainties:
                    entry += f" [Uncertain: {output.uncertainties[0][:30]}]"
                per_persona_history[persona_name].append(entry)
                if cross_visible:
                    discussion_history.append(entry)

                self.personality.step(f"Turn {turn + 1} - {persona_name}: Processing...")

            # Convergence check: persistent tensions are useful metadata,
            # but the tension records name other personas ("Linguistic became
            # less certain..."). Adding them to cognitive_state means each
            # persona's NEXT turn would serialize a cognitive_state containing
            # peer names into its own memory metadata - a stored-state leak.
            #
            # When isolation is on (default), we skip the cross-persona
            # accumulation. The Composer can still synthesize tension analysis
            # directly from final_outputs / specialist_outputs.
            if turn > 0 and cross_visible:
                tensions = self._identify_persistent_tensions(position_evolution)
                if tensions:
                    cognitive_state.unresolved_tensions.extend(tensions)

        # Final refined analysis pass. Each persona sees only its own history
        # (or the global one if cross-visibility is enabled) and the user query.
        self.personality.step("Generating final refined analyses...")

        final_outputs = {}
        for name in personas_to_activate:
            own_hist = per_persona_history.get(name, [])
            if cross_visible:
                history_for_prompt = discussion_history[-5:]
            else:
                history_for_prompt = own_hist[-5:]

            final_context = f"""After your {self.operation_mode.value} reflection, provide your final analysis.
Your refinement trajectory:
{chr(10).join(history_for_prompt)}

Unresolved tensions on this question: {len(cognitive_state.unresolved_tensions)}
Express any remaining uncertainties or curiosities."""

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

                explanation_output = persona.process_query(
                    meta_prompt,
                    cognitive_state=cognitive_state,
                    is_meta_introspection=True,  # Tag so this doesn't poison analytical RAG
                )
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
{chr(10).join([f"{e.persona_name}: {e.response[:ATHENA_RESPONSE_PASSTHROUGH_MAX_CHARS]}" for e in explanations])}

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
            
            # Mark today's prep fragments as staged_for_training so anything
            # reading the dream_fragments table can tell which payloads have been
            # exported. Done before warning about the LoRA placeholder so even
            # without training, the state machine advances correctly.
            for name, persona in self.specialist_personas.items():
                try:
                    persona.memory.mark_dream_fragments(
                        fragment_type='consolidation_prep',
                        new_status='staged_for_training',
                    )
                except Exception as e:
                    self.app.warning(f"Could not mark {name} fragments as staged: {e}")
            try:
                self.composer.memory.mark_dream_fragments(
                    fragment_type='consolidation_prep',
                    new_status='staged_for_training',
                )
            except Exception as e:
                self.app.warning(f"Could not mark composer fragments as staged: {e}")

            self.app.success("=== SLEEP CYCLE PREPARATION COMPLETE ===")
            # TODO(future-hardware): Wire up the LoRA training pipeline here once GPU resources allow.
            # The JSON files written above are the staging payload for that future trainer.
            self.app.warning("LoRA training implementation pending - data prepared for external training")

            # Placeholder for actual LoRA training
            self.app.info("To implement LoRA training:")
            self.app.info("1. Use prepared JSON files for each persona")
            self.app.info("2. Train individual LoRA adapters")
            self.app.info("3. Monthly merge into base model")
            
        except Exception as e:
            trace_exception(e)
            self.app.error(f"Sleep cycle failed: {e}")

    def _extract_recent_conversation(
        self,
        context: LollmsContextDetails,
        max_turns: int = ATHENA_CONVERSATION_MEMORY_TURNS,
    ) -> List[Dict[str, str]]:
        """Pull the last `max_turns` user/assistant exchanges from the lollms context.

        Returns a list of {sender, content} dicts in chronological order. Used to give
        the Composer actual conversation memory so it can say "as we just discussed"
        rather than treating each query as a cold start.

        Defensive: lollms message shapes vary; we try multiple known forms and
        return [] on any extraction failure rather than raising.
        """
        try:
            messages = getattr(context, "discussion_messages", None)
            if not messages:
                return []

            collected: List[Dict[str, str]] = []
            # Walk newest-first so we pick the most recent turns then reverse.
            for message in reversed(messages):
                if not isinstance(message, dict):
                    # String-form messages: skip - they don't carry sender info reliably.
                    continue

                content = (
                    message.get("content")
                    or message.get("message")
                    or message.get("text")
                    or ""
                )
                content = str(content).strip()
                if not content:
                    continue

                # Multiple shape variants for who sent it:
                #  - {"type": 0} for user, {"type": 1} for assistant (lollms classic)
                #  - {"sender": "user"|"assistant"|"system"}
                #  - {"role": "user"|"assistant"|"system"}
                sender = "unknown"
                if "sender" in message:
                    sender = str(message["sender"]).lower()
                elif "role" in message:
                    sender = str(message["role"]).lower()
                elif "type" in message:
                    sender = "user" if message["type"] == 0 else "assistant"

                # Skip system messages - they're typically prompts, not real turns.
                if sender in ("system", "tool"):
                    continue

                collected.append({"sender": sender, "content": content})
                # Stop once we have enough turns. We count by message, not by
                # user/assistant pair, since some conversations have multiple
                # consecutive messages from one side.
                if len(collected) >= max_turns * 2:
                    break

            # Drop the very last user message (that's the current query, the
            # system is already processing it - including it would be redundant).
            if collected and collected[0].get("sender") == "user":
                collected = collected[1:]

            collected.reverse()
            return collected
        except Exception as e:
            # Non-fatal: synthesis proceeds without conversation memory.
            self.app.warning(f"Conversation memory extraction failed (non-fatal): {e}")
            return []

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
            # Use a hash of the full normalized query as the cache key. Previous
            # implementation used query[:50] which caused collisions: two long
            # queries sharing a 50-char prefix would reuse each other's routing
            # decisions and cognitive state - a real correctness bug.
            query_normalized = ' '.join(query.lower().split())  # collapse whitespace
            query_prefix = hashlib.blake2b(query_normalized.encode('utf-8'), digest_size=16).hexdigest()
            cached_cognitive_state = getattr(self, '_cognitive_cache', {}).get(query_prefix)
        
            if cached_cognitive_state and hasattr(cached_cognitive_state, 'timestamp'):
                # Use cache if less than 5 minutes old
                if (datetime.now() - cached_cognitive_state.timestamp).total_seconds() < ATHENA_COGNITIVE_CACHE_TTL_SECONDS:
                    self.cognitive_state = cached_cognitive_state
                    self.app.info("Using cached cognitive state")
                else:
                    cached_cognitive_state = None
        
            # Check for special commands

            # User feedback hook: queries starting with "feedback:" are routed
            # to the error_autobiography pipeline so the system can learn from
            # explicit user signal. Format examples:
            #   feedback: positive - that was a really helpful answer
            #   feedback: negative - the answer missed the main point
            #   feedback: that was incorrect about Python's GIL
            feedback_match = re.match(
                r'^\s*feedback\s*[:\-]\s*(positive|negative|neutral)?\s*[:\-]?\s*(.*)',
                query, re.IGNORECASE,
            )
            if feedback_match:
                tone = (feedback_match.group(1) or "neutral").lower()
                comment = (feedback_match.group(2) or "").strip()
                severity = {"negative": 0.8, "neutral": 0.4, "positive": 0.1}.get(tone, 0.4)

                # Find the most recent Composer synthesis to attach feedback to.
                recent = self.composer.memory.retrieve_memories(limit=3)
                target = recent[0] if recent else None
                target_query = target.query if target else "(no recent response)"
                target_response = target.response if target else ""

                # Record on the Composer's memory.
                self.composer.memory.record_error(
                    query=target_query,
                    incorrect=target_response if tone == "negative" else "(positive feedback)",
                    correction=comment if tone == "negative" else "",
                    reflection=f"User feedback ({tone}): {comment}",
                    error_type=f"user_feedback_{tone}",
                    severity=severity,
                )

                constructed_context.clear()
                constructed_context.append("<!-- ATHENA_FEEDBACK -->")
                emoji = {"positive": "🟢", "negative": "🟡", "neutral": "⚪"}.get(tone, "⚪")
                constructed_context.append(
                    f"\n## {emoji} Feedback recorded\n\n"
                    f"Tone: **{tone}**\n\n"
                    f"Comment: {comment if comment else '(none)'}\n\n"
                    f"Logged to error autobiography. Future LoRA consolidation "
                    f"will weight this signal."
                )
                return constructed_context

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
                    if len(self._cognitive_cache) > ATHENA_COGNITIVE_CACHE_SIZE:
                        # Remove oldest entries
                        oldest_keys = list(self._cognitive_cache.keys())[:ATHENA_COGNITIVE_CACHE_SIZE // 2]
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
                        # Standard sequential processing.
                        #
                        # COGNITIVE ISOLATION (default): each specialist analyzes the user query
                        # independently and never sees prior specialists' responses. The Composer
                        # is the ONLY component that sees the full set of SpecialistOutput objects.
                        # This preserves the diversity-of-perspective premise of the architecture -
                        # if one persona reads another's response first, anchoring kicks in and the
                        # "independent" view is no longer independent.
                        #
                        # Legacy chained behavior is still available via the config flag
                        # `enable_persona_cross_visibility` for users who want it.
                        cross_visible = self.static_parameters.config.get("enable_persona_cross_visibility", False)
                        chained_context = ""  # Only grown when cross_visible is True
                        for name in personas_to_activate:
                            if name in self.specialist_personas:
                                persona = self.specialist_personas[name]

                                # Ensure we're passing the actual query, not empty string
                                if not query or query == "..":
                                    self.app.error(f"Invalid query being passed to {name}: '{query}'")
                                    query = "Hello"  # Emergency fallback

                                # In isolated mode (default) shared_context is None - the persona
                                # only sees the user query plus its OWN persisted memory.
                                output = persona.process_query(
                                    query,
                                    shared_context=(chained_context if cross_visible else None),
                                    cognitive_state=self.cognitive_state
                                )
                                specialist_outputs.append(output)

                                if cross_visible:
                                    # Legacy path: accumulate so the next persona can see prior work.
                                    chained_context += f"\n--- Analysis from {name} ---\n{output.response}\n"
                                    if len(chained_context) > ATHENA_CHAINED_CONTEXT_MAX_CHARS:
                                        chained_context = chained_context[-ATHENA_CHAINED_CONTEXT_KEEP_CHARS:]
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
            
                # 7. Synthesize with confusion awareness.
                # Pull conversation memory and pass to the Composer so it can
                # reference what was just discussed; pull self-RAG flag from
                # config so it can be turned off when desired.
                conversation_history = []
                if self.static_parameters.config.get("enable_conversation_memory", True):
                    conversation_history = self._extract_recent_conversation(context)
                enable_self_rag = self.static_parameters.config.get("enable_composer_self_rag", True)

                if specialist_outputs:
                    final_output = self.composer.synthesize(
                        query, specialist_outputs, weights,
                        self.cognitive_state, self.output_format,
                        stream_thoughts,
                        conversation_history=conversation_history,
                        enable_self_rag=enable_self_rag,
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
            if hasattr(self, 'cognitive_state') and self.cognitive_state.confusion_level > ATHENA_CONFUSION_HIGH_BANNER:
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
        
            # Add routing transparency footer if enabled. Shows which personas
            # were activated and why - useful for paper screenshots and for users
            # who want to understand how the system arrived at its routing.
            if (self.static_parameters.config.get("enable_routing_transparency", False)
                    and hasattr(self, 'cognitive_state')
                    and self.cognitive_state
                    and self.cognitive_state.routing_explanation):
                output_parts.append(
                    f"\n\n*[Routing: {self.cognitive_state.routing_explanation}]*"
                )

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
        """Cleanup on deletion.

        Wrapped in try/except because exceptions raised from __del__ are unsupported
        by the interpreter and produce noisy warnings during shutdown. We also use
        getattr-with-default rather than hasattr to be safe against partial init.
        """
        try:
            soc = getattr(self, 'stream_of_consciousness', None)
            if soc is not None:
                soc.stop()
        except Exception:
            pass
        # v3: best-effort unload of per-persona models so VRAM is released cleanly.
        try:
            for persona in getattr(self, 'specialist_personas', {}).values():
                client = getattr(persona, '_own_client', None)
                mp = getattr(persona, 'model_path', None)
                if client is not None and mp:
                    try:
                        client.llm.unload_model(mp)
                    except Exception:
                        pass
        except Exception:
            pass