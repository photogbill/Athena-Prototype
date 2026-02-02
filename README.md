# Project ATHENA 🏛️
### Architecture for Theoretically Holistic Expert Networked Analysis

![Version](https://img.shields.io/badge/version-2.2-blue) ![Status](https://img.shields.io/badge/status-prototype-orange) ![Framework](https://img.shields.io/badge/framework-LoLLMs-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Project ATHENA** is a multi-persona cognitive architecture that operationalizes Howard Gardner's Theory of Multiple Intelligences within a decentralized AI system.

Unlike monolithic LLM interactions, ATHENA deconstructs queries into a society of eight specialized personas, each representing a distinct cognitive domain (e.g., Logical-Mathematical, Interpersonal, Spatial). These isolated perspectives are routed by an Orchestrator and synthesized by a Composer into a coherent, psychologically grounded response.

---

## 🧠 Key Features

### 1. Multi-Persona Cognitive Isolation
* **The Society of Mind:** Eight specialized personas operate with strict cognitive isolation, utilizing unique temperature settings and reasoning patterns tailored to their domain.
* **Dynamic Routing:** An **Orchestrator** analyzes incoming queries for complexity, emotional context, and urgency to activate only the most relevant intelligences.

### 2. Mixture of Reasoning (MoR)
ATHENA does not rely on a single reasoning mode. Each persona simultaneously applies:
* **Deductive & Inductive Logic**
* **Abductive Inference**
* **Analogical Mapping**
* **Dialectical Synthesis**

### 3. Advanced Memory Architecture
A sophisticated SQLite-based memory system capable of:
* **Belief Tension Tracking:** Identifies and tags conflicts between new insights and past memories.
* **Error Autobiography:** A self-correction system that stores, categorizes, and learns from mistakes.
* **Dream Consolidation:** A sleep-cycle mechanism that abstracts daily interactions into high-level patterns for long-term learning (prepared for QLoRA fine-tuning).

### 4. Constitutional AI & Ethics
* **The Relationship Paradigm:** Rather than external guardrails, the **Constitutional Persona** acts as an integrated ethical intelligence, participating in the cognitive process to evolve principles based on interaction history.

### 5. Stream of Consciousness
* **Background Processing:** An idle-state cognitive loop that generates thoughts, questions assumptions, and connects disparate ideas when the system is not actively processing user queries.

---

## 🛠️ Installation

Project ATHENA is built as a function extension for the **LoLLMs (Lord of Large Language and Multimodal Systems)** framework.

### Prerequisites
* [LoLLMs WebUI](https://github.com/ParisNeo/lollms-webui) installed and running.
* Python 3.10+

### Setup
1.  Navigate to your LoLLMs extension directory:
    ```bash
    cd /path/to/lollms-webui/functions/
    ```
2.  Clone this repository or create a folder named `athena_prototype`:
    ```bash
    git clone [https://github.com/photogbill/Athena-Prototype.git](https://github.com/photogbill/Athena-Prototype.git) athena_prototype
    ```
3.  Restart your LoLLMs instance.
4.  Enable **Project ATHENA** in the active functions list within the LoLLMs settings.

---

## ⚙️ Configuration

ATHENA is highly configurable. You can adjust the following parameters in the LoLLMs function settings or directly in `config.yaml`:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `operation_mode` | `standard` | `standard`, `collaborative`, or `adversarial` (debate mode). |
| `final_output_format` | `visual_dialogue` | Style of the final response (e.g., `screenplay`, `mind_map`, `formal_transcript`). |
| `enable_stream_of_consciousness` | `False` | Enables background thought generation between queries. |
| `enable_dream_consolidation` | `True` | Activates the sleep-cycle pattern extraction system. |
| `enable_constitutional_persona` | `True` | Enables the evolving ethical oversight layer. |

---

## 🚀 Usage

Once enabled, ATHENA automatically intercepts and processes your queries through the Orchestrator.

### Standard Interaction
Simply chat with the AI. ATHENA will silently route your query to the relevant specialists (e.g., asking for code triggers *Logical-Mathematical* and *Linguistic*; asking for relationship advice triggers *Interpersonal* and *Intrapersonal*).

### Special Commands
You can trigger specific cognitive workflows using these keywords:

* **"Explain yourself"** / **"Show your work"**: Triggers the **Explainability Workflow**, revealing the internal chain-of-thought and which personas contributed to the answer.
* **"Trigger sleep cycle"**: Initiates **Dream Consolidation**, compressing the session's memories into abstract patterns and JSON data for training.
* **"What are you thinking?"**: Accesses the **Stream of Consciousness** buffer to see what the AI has been "musing" about in the background.

---

## 📐 Architecture Overview

![Athena Architecture Diagram](https://github.com/photogbill/Athena-Prototype/blob/main/Athena_Mermaid_Diagram_Full.svg?raw=true)

---

## 📄 Documentation

* [**White Paper v2.2**](./docs/White_Paper_ATHENA_v2_2.pdf): Complete technical specification, philosophical foundation, and algorithm details.
* [**arXiv Preprint**](./docs/athena_arxiv_paper.pdf): Academic paper regarding the architecture.

---

## 🤝 Contributing

Contributions are welcome! Please focus on:
1.  **New Persona Modules:** specialized prompts for new domains.
2.  **Memory Optimizations:** Improving the vector storage efficiency.
3.  **Visualization Tools:** Front-end components to visualize the active personas in real-time.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*Author: William R. Duncan*
