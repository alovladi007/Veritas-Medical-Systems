"""
KGAREVION Core Pipeline Implementation
Medical QA System with Knowledge Graph Verification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
import logging
from neo4j import AsyncGraphDatabase
import redis.asyncio as redis
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Data Models
@dataclass
class MedicalTriplet:
    """Represents a medical knowledge triplet"""
    head: str
    relation: str
    tail: str
    confidence: float = 1.0
    verified: bool = False
    source: str = "generated"


@dataclass
class Question:
    """Medical question structure"""
    text: str
    question_type: str  # "multiple_choice" or "open_ended"
    candidates: Optional[List[str]] = None
    medical_entities: Optional[List[str]] = None


class RelationType(Enum):
    """Medical relation types from the paper"""
    PROTEIN_PROTEIN = "protein_protein"
    CARRIER = "carrier"
    ENZYME = "enzyme"
    TARGET = "target"
    TRANSPORTER = "transporter"
    CONTRAINDICATION = "contraindication"
    INDICATION = "indication"
    OFF_LABEL = "off_label_use"
    SYNERGISTIC = "synergistic_interaction"
    ASSOCIATED = "associated_with"
    PARENT_CHILD = "parent_child"
    PHENOTYPE_ABSENT = "phenotype_absent"
    PHENOTYPE_PRESENT = "phenotype_present"
    SIDE_EFFECT = "side_effect"
    INTERACTS = "interacts_with"
    LINKED = "linked_to"
    EXPRESSION_PRESENT = "expression_present"
    EXPRESSION_ABSENT = "expression_absent"


# Embedding Alignment Module (from paper Section 3.2)
class EmbeddingAlignmentModule(nn.Module):
    """Aligns structural KG embeddings with LLM token embeddings"""
    
    def __init__(self, kg_dim: int = 128, llm_dim: int = 4096, hidden_dim: int = 2048):
        super().__init__()
        self.projection = nn.Linear(kg_dim, llm_dim)
        self.attention = nn.MultiheadAttention(llm_dim, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.LayerNorm(llm_dim),
            nn.Linear(llm_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, llm_dim)
        )
        
    def forward(self, kg_embeddings: torch.Tensor, token_embeddings: torch.Tensor):
        # Project KG embeddings to LLM dimension
        kg_proj = self.projection(kg_embeddings)
        
        # Attention mechanism (Equation 3 from paper)
        attended, _ = self.attention(kg_proj, token_embeddings, token_embeddings)
        aligned = kg_proj + attended
        
        # Feed-forward network (Equation 4)
        output = aligned + self.ffn(aligned)
        return output


class KnowledgeGraphInterface:
    """Interface for Neo4j knowledge graph operations"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.entity_embeddings = {}  # Cache for TransE embeddings
        
    async def close(self):
        await self.driver.close()
        
    async def map_entity_to_umls(self, entity: str) -> Optional[str]:
        """Map entity to UMLS code"""
        async with self.driver.session() as session:
            query = """
            MATCH (n:Entity {name: $entity})
            RETURN n.umls_code as umls_code
            """
            result = await session.run(query, entity=entity)
            record = await result.single()
            return record["umls_code"] if record else None
            
    async def get_entity_embedding(self, entity: str) -> Optional[np.ndarray]:
        """Get pre-computed TransE embedding for entity"""
        if entity in self.entity_embeddings:
            return self.entity_embeddings[entity]
            
        async with self.driver.session() as session:
            query = """
            MATCH (n:Entity {name: $entity})
            RETURN n.embedding as embedding
            """
            result = await session.run(query, entity=entity)
            record = await result.single()
            if record and record["embedding"]:
                embedding = np.array(record["embedding"])
                self.entity_embeddings[entity] = embedding
                return embedding
        return None
        
    async def verify_triplet(self, triplet: MedicalTriplet) -> bool:
        """Verify if triplet exists in KG"""
        async with self.driver.session() as session:
            query = """
            MATCH (h:Entity {name: $head})-[r:$relation]->(t:Entity {name: $tail})
            RETURN COUNT(*) > 0 as exists
            """
            result = await session.run(
                query,
                head=triplet.head,
                relation=triplet.relation,
                tail=triplet.tail
            )
            record = await result.single()
            return record["exists"] if record else False


class MedicalEntityExtractor:
    """Extract medical entities from text"""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"):
        self.model = SentenceTransformer(model_name)
        self.medical_terms_cache = set()
        
    def extract_entities(self, text: str) -> List[str]:
        """Extract medical entities from question text"""
        # Simple extraction - in production, use BioBERT NER
        entities = []
        
        # Common medical entity patterns
        patterns = [
            r'\b[A-Z][A-Z0-9]{2,}\b',  # Protein/gene names
            r'\b\w+itis\b',  # Diseases ending in -itis
            r'\b\w+osis\b',  # Diseases ending in -osis
            r'\b\w+oma\b',   # Tumors
        ]
        
        import re
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
            
        # Remove duplicates and return
        return list(set(entities))


class GenerateAction:
    """Generate triplets from medical questions"""
    
    def __init__(self, llm_model: str = "meta-llama/Llama-3-8B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto"
        )
        self.entity_extractor = MedicalEntityExtractor()
        
    def generate_triplets(self, question: Question) -> List[MedicalTriplet]:
        """Generate relevant triplets based on question type"""
        triplets = []
        
        # Extract medical entities
        entities = self.entity_extractor.extract_entities(question.text)
        question.medical_entities = entities
        
        if question.question_type == "multiple_choice" and question.candidates:
            # Choice-aware generation (Equation 1 from paper)
            for candidate in question.candidates:
                candidate_triplets = self._generate_for_candidate(question, candidate)
                triplets.extend(candidate_triplets)
        else:
            # Non-choice-aware generation
            triplets = self._generate_from_entities(question, entities)
            
        return triplets
        
    def _generate_for_candidate(self, question: Question, candidate: str) -> List[MedicalTriplet]:
        """Generate triplets for a specific answer candidate"""
        prompt = f"""### Instruction:
Given the following question and answer candidate, generate relevant medical triplets.

### Input:
Question: {question.text}
Candidate: {candidate}
Medical Entities: {', '.join(question.medical_entities or [])}

### Response:
Generate triplets in format (head, relation, tail):
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.7,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_triplets(response)
        
    def _generate_from_entities(self, question: Question, entities: List[str]) -> List[MedicalTriplet]:
        """Generate triplets from extracted entities"""
        triplets = []
        # Generate pairwise relationships between entities
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                # Use LLM to determine relationship
                triplet = self._infer_relationship(entity1, entity2, question.text)
                if triplet:
                    triplets.append(triplet)
        return triplets
        
    def _infer_relationship(self, entity1: str, entity2: str, context: str) -> Optional[MedicalTriplet]:
        """Infer relationship between two entities"""
        prompt = f"""Determine the medical relationship between {entity1} and {entity2} 
        in the context: {context}
        
        Relationship must be one of: {[r.value for r in RelationType]}
        """
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Parse response and create triplet
        # Simplified - implement proper parsing
        return MedicalTriplet(entity1, "associated_with", entity2)
        
    def _parse_triplets(self, response: str) -> List[MedicalTriplet]:
        """Parse LLM response into triplet objects"""
        triplets = []
        # Simple parsing - improve with structured generation
        lines = response.split('\n')
        for line in lines:
            if '(' in line and ')' in line:
                try:
                    # Extract triplet from format (head, relation, tail)
                    content = line[line.index('(')+1:line.index(')')]
                    parts = [p.strip() for p in content.split(',')]
                    if len(parts) == 3:
                        triplets.append(MedicalTriplet(*parts))
                except:
                    continue
        return triplets


class ReviewAction:
    """Review and verify generated triplets using KG"""
    
    def __init__(self, llm_model: str, kg_interface: KnowledgeGraphInterface):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto"
        )
        self.kg = kg_interface
        self.alignment_module = EmbeddingAlignmentModule()
        
        # Configure LoRA for fine-tuning
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        self.model = get_peft_model(self.base_model, lora_config)
        
    async def review_triplets(self, triplets: List[MedicalTriplet]) -> Tuple[List[MedicalTriplet], List[MedicalTriplet]]:
        """Review triplets and separate into verified and false sets"""
        verified_triplets = []
        false_triplets = []
        
        for triplet in triplets:
            is_valid = await self._verify_single_triplet(triplet)
            if is_valid:
                triplet.verified = True
                verified_triplets.append(triplet)
            else:
                false_triplets.append(triplet)
                
        return verified_triplets, false_triplets
        
    async def _verify_single_triplet(self, triplet: MedicalTriplet) -> bool:
        """Verify a single triplet using fine-tuned model and KG"""
        # Get embeddings from KG
        head_emb = await self.kg.get_entity_embedding(triplet.head)
        tail_emb = await self.kg.get_entity_embedding(triplet.tail)
        
        # Apply soft constraint rules from paper
        if head_emb is None or tail_emb is None:
            # Incomplete knowledge - keep triplet
            return True
            
        # Check if triplet exists in KG
        exists_in_kg = await self.kg.verify_triplet(triplet)
        if exists_in_kg:
            return True
            
        # Use fine-tuned model for verification
        prompt = f"""### Instruction:
Given a triple from a knowledge graph, determine if it is correct.

### Input:
Triple: ({triplet.head}, {triplet.relation}, {triplet.tail})

### Response:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Align embeddings if available
        if head_emb is not None and tail_emb is not None:
            kg_embeddings = torch.tensor(np.stack([head_emb, tail_emb]))
            token_embeddings = self.base_model.get_input_embeddings()(inputs.input_ids)
            aligned_embeddings = self.alignment_module(kg_embeddings.unsqueeze(0), token_embeddings)
            
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=10)
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return "true" in response.lower()


class ReviseAction:
    """Revise false triplets to correct them"""
    
    def __init__(self, llm_model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto"
        )
        
    def revise_triplets(self, false_triplets: List[MedicalTriplet], question: Question) -> List[MedicalTriplet]:
        """Revise false triplets to make them correct"""
        revised_triplets = []
        
        for triplet in false_triplets:
            revised = self._revise_single_triplet(triplet, question)
            if revised:
                revised_triplets.append(revised)
                
        return revised_triplets
        
    def _revise_single_triplet(self, triplet: MedicalTriplet, question: Question) -> Optional[MedicalTriplet]:
        """Revise a single triplet"""
        prompt = f"""### Instruction:
Revise the following medical triplet to make it accurate for answering the question.

### Input:
Question: {question.text}
Incorrect Triplet: ({triplet.head}, {triplet.relation}, {triplet.tail})

### Response:
Revised Triplet:"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=100, temperature=0.7)
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Parse revised triplet
        # Simplified parsing - implement proper extraction
        try:
            if '(' in response and ')' in response:
                content = response[response.rindex('(')+1:response.rindex(')')]
                parts = [p.strip() for p in content.split(',')]
                if len(parts) == 3:
                    return MedicalTriplet(*parts, source="revised")
        except:
            pass
            
        return None


class AnswerAction:
    """Generate final answer from verified triplets"""
    
    def __init__(self, llm_model: str):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto"
        )
        
    def generate_answer(self, question: Question, verified_triplets: List[MedicalTriplet]) -> str:
        """Generate final answer based on verified triplets"""
        # Format triplets for prompt
        triplet_text = "\n".join([
            f"- {t.head} {t.relation} {t.tail}" for t in verified_triplets
        ])
        
        if question.question_type == "multiple_choice":
            prompt = f"""### Instruction:
Based on the following verified medical knowledge, select the correct answer.

### Knowledge:
{triplet_text}

### Question:
{question.text}

### Options:
{chr(10).join([f"{chr(65+i)}: {c}" for i, c in enumerate(question.candidates or [])])}

### Answer:"""
        else:
            prompt = f"""### Instruction:
Based on the following verified medical knowledge, answer the question.

### Knowledge:
{triplet_text}

### Question:
{question.text}

### Answer:"""
            
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                temperature=0.3,  # Lower temperature for more focused answers
                do_sample=True
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract answer from response
        answer_start = response.find("### Answer:") + len("### Answer:")
        answer = response[answer_start:].strip()
        
        return answer


class KGARevionPipeline:
    """Main KGAREVION pipeline orchestrator"""
    
    def __init__(
        self,
        llm_model: str = "meta-llama/Llama-3-8B-Instruct",
        kg_uri: str = "neo4j://localhost:7687",
        kg_user: str = "neo4j",
        kg_password: str = "password",
        redis_host: str = "localhost",
        redis_port: int = 6379
    ):
        # Initialize components
        self.kg = KnowledgeGraphInterface(kg_uri, kg_user, kg_password)
        self.cache = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        
        self.generate_action = GenerateAction(llm_model)
        self.review_action = ReviewAction(llm_model, self.kg)
        self.revise_action = ReviseAction(llm_model)
        self.answer_action = AnswerAction(llm_model)
        
        self.max_revise_rounds = 3
        
    async def process_question(self, question: Question) -> Dict[str, Any]:
        """Process a medical question through the full pipeline"""
        
        # Check cache first
        cache_key = f"qa:{hash(question.text)}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
            
        logger.info(f"Processing question: {question.text[:100]}...")
        
        # Step 1: Generate triplets
        generated_triplets = self.generate_action.generate_triplets(question)
        logger.info(f"Generated {len(generated_triplets)} triplets")
        
        # Step 2: Review triplets
        verified_triplets, false_triplets = await self.review_action.review_triplets(generated_triplets)
        logger.info(f"Verified: {len(verified_triplets)}, False: {len(false_triplets)}")
        
        # Step 3: Revise false triplets (iteratively)
        for round_num in range(self.max_revise_rounds):
            if not false_triplets:
                break
                
            logger.info(f"Revise round {round_num + 1}")
            revised_triplets = self.revise_action.revise_triplets(false_triplets, question)
            
            # Re-review revised triplets
            newly_verified, still_false = await self.review_action.review_triplets(revised_triplets)
            verified_triplets.extend(newly_verified)
            false_triplets = still_false
            
        # Step 4: Generate answer
        answer = self.answer_action.generate_answer(question, verified_triplets)
        
        result = {
            "question": question.text,
            "answer": answer,
            "verified_triplets": [
                {
                    "head": t.head,
                    "relation": t.relation, 
                    "tail": t.tail,
                    "confidence": t.confidence
                }
                for t in verified_triplets
            ],
            "medical_entities": question.medical_entities,
            "processing_stats": {
                "total_triplets_generated": len(generated_triplets),
                "triplets_verified": len(verified_triplets),
                "triplets_revised": len(generated_triplets) - len(verified_triplets) - len(false_triplets)
            }
        }
        
        # Cache result
        await self.cache.setex(cache_key, 3600, json.dumps(result))
        
        return result
        
    async def close(self):
        """Clean up resources"""
        await self.kg.close()
        await self.cache.close()


# Example usage
async def main():
    # Initialize pipeline
    pipeline = KGARevionPipeline(
        llm_model="meta-llama/Llama-3-8B-Instruct",
        kg_uri="neo4j://localhost:7687",
        kg_user="neo4j",
        kg_password="your_password"
    )
    
    # Example multiple choice question
    question = Question(
        text="Which protein is associated with Retinitis Pigmentosa 59?",
        question_type="multiple_choice",
        candidates=["HSPA4", "HSPA8", "HSPA1B", "HSPA1A"]
    )
    
    # Process question
    result = await pipeline.process_question(question)
    print(json.dumps(result, indent=2))
    
    # Clean up
    await pipeline.close()


if __name__ == "__main__":
    asyncio.run(main())
