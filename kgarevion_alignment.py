"""
KGAREVION Embedding Alignment Implementation
Proper prefix token integration and description dictionary for bridging 
semantic and structural representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from peft import LoraConfig, get_peft_model, TaskType, PeftModel


# Description Dictionary - Maps relations to semantic descriptions
# Based on Table 8 from the paper's appendix
RELATION_DESCRIPTIONS = {
    "protein_protein": "{A} interacts with {B}, indicating that the two proteins directly or indirectly associate with each other to perform a biological function",
    "carrier": "{A} acts as a carrier for {B}, facilitating its transport or delivery to specific locations within the body or within a cell",
    "enzyme": "{A} functions as an enzyme that catalyzes a reaction involving {B}, converting it into a different molecule or modifying its structure",
    "target": "{A} serves as a target for {B}, meaning that {B} binds to or interacts with {A} to exert its biological effect",
    "transporter": "{A} is a transporter that facilitates the movement of {B} across cellular membranes or within different compartments of the body",
    "contraindication": "The interaction between {A} and {B} is contraindicated, meaning that the presence of one molecule may have adverse effects or reduce the efficacy of the other",
    "indication": "{A} is indicated for the treatment or management of a condition associated with {B}, suggesting that {A} has a therapeutic role related to {B}",
    "off_label_use": "{A} is used off-label in relation to {B}, meaning it is utilized in a manner not specifically approved but based on clinical judgment",
    "synergistic_interaction": "{A} and {B} interact synergistically, where their combined effect is greater than the sum of their individual effects",
    "associated_with": "{A} is associated with {B}, indicating a relationship or correlation between the two, often in the context of disease or biological processes",
    "parent_child": "{A} is related to {B} in a parent-child relationship, where {A} gives rise to or influences the formation of {B}",
    "phenotype_absent": "The interaction between {A} and {B} results in the absence of a specific phenotype, indicating that the normal trait is not expressed",
    "phenotype_present": "The interaction between {A} and {B} results in the presence of a specific phenotype, indicating that a particular trait is expressed",
    "side_effect": "The interaction between {A} and {B} can cause a side effect, where the presence of one molecule leads to unintended and possibly adverse effects",
    "interacts_with": "{A} interacts with {B}, indicating a general interaction that may involve binding, modulation, or other forms of molecular communication",
    "linked_to": "{A} is linked to {B}, suggesting a connection or association between the two molecules, often in a biological or pathological context",
    "expression_present": "{A} is expressed in the presence of {B}, indicating that the existence or activity of {B} leads to or correlates with the expression of {A}",
    "expression_absent": "{A} is not expressed in the presence of {B}, indicating that the existence or activity of {B} suppresses or does not correlate with the expression of {A}"
}


@dataclass
class AlignedTriplet:
    """Stores triplet with aligned embeddings"""
    head: str
    relation: str
    tail: str
    head_kg_emb: torch.Tensor  # KG embedding from TransE
    relation_kg_emb: torch.Tensor
    tail_kg_emb: torch.Tensor
    aligned_prefix_tokens: torch.Tensor  # Aligned embeddings as prefix tokens
    description: str


class KGtoLLMAlignmentModule(nn.Module):
    """
    Aligns KG structural embeddings with LLM semantic space
    Following Section 3.2 of the paper
    """
    
    def __init__(
        self,
        kg_embedding_dim: int = 128,  # TransE embedding dimension
        llm_hidden_dim: int = 4096,   # LLM hidden dimension (e.g., LLaMA-3)
        num_prefix_tokens: int = 3,    # Number of prefix tokens per triplet
        llm_model_name: str = "meta-llama/Llama-3-8B-Instruct"
    ):
        super().__init__()
        
        self.kg_embedding_dim = kg_embedding_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.num_prefix_tokens = num_prefix_tokens
        
        # Linear projection from KG space to LLM space (g(·) in paper)
        self.kg_projection = nn.Linear(kg_embedding_dim, llm_hidden_dim)
        
        # Multi-head attention for alignment (Equation 3)
        self.alignment_attention = nn.MultiheadAttention(
            embed_dim=llm_hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Feed-forward network (Equation 4)
        self.ffn = nn.Sequential(
            nn.LayerNorm(llm_hidden_dim),
            nn.Linear(llm_hidden_dim, llm_hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(llm_hidden_dim * 2, llm_hidden_dim)
        )
        
        # Initialize tokenizer for description processing
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def generate_triplet_description(self, head: str, relation: str, tail: str) -> str:
        """
        Generate semantic description for triplet using description dictionary
        """
        if relation in RELATION_DESCRIPTIONS:
            description = RELATION_DESCRIPTIONS[relation]
            # Replace placeholders with actual entities
            description = description.replace("{A}", head).replace("{B}", tail)
        else:
            # Fallback for unknown relations
            description = f"{head} has relation '{relation}' with {tail}"
        
        return description
    
    def align_embeddings(
        self,
        kg_embeddings: torch.Tensor,  # Shape: (batch, 3, kg_dim) for head, rel, tail
        description_embeddings: torch.Tensor  # Shape: (batch, seq_len, llm_dim)
    ) -> torch.Tensor:
        """
        Align KG structural embeddings with LLM token embeddings
        Returns prefix tokens for LLM
        """
        batch_size = kg_embeddings.size(0)
        
        # Project KG embeddings to LLM dimension [batch, 3, llm_dim]
        kg_projected = self.kg_projection(kg_embeddings)
        
        # Apply attention between KG and description embeddings (Equation 3)
        # Query: KG embeddings, Key/Value: Description embeddings
        aligned_embeddings, _ = self.alignment_attention(
            query=kg_projected,
            key=description_embeddings,
            value=description_embeddings
        )
        
        # Add residual connection
        aligned_embeddings = kg_projected + aligned_embeddings
        
        # Apply FFN (Equation 4)
        final_embeddings = aligned_embeddings + self.ffn(aligned_embeddings)
        
        # Shape: [batch, 3, llm_dim] - 3 prefix tokens per triplet
        return final_embeddings
    
    def prepare_prefix_tokens(
        self,
        triplet: Tuple[str, str, str],
        kg_embeddings: Tuple[np.ndarray, np.ndarray, np.ndarray],
        llm_model: nn.Module
    ) -> AlignedTriplet:
        """
        Prepare aligned prefix tokens for a single triplet
        This is what gets prepended to the LLM input during inference
        """
        head, relation, tail = triplet
        head_emb, rel_emb, tail_emb = kg_embeddings
        
        # Generate description
        description = self.generate_triplet_description(head, relation, tail)
        
        # Tokenize description
        description_tokens = self.tokenizer(
            description,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Get token embeddings from LLM
        with torch.no_grad():
            token_embeddings = llm_model.get_input_embeddings()(description_tokens["input_ids"])
        
        # Convert KG embeddings to tensor
        kg_emb_tensor = torch.tensor(
            np.stack([head_emb, rel_emb, tail_emb]),
            dtype=torch.float32
        ).unsqueeze(0)  # Add batch dimension
        
        # Align embeddings
        aligned_prefix = self.align_embeddings(kg_emb_tensor, token_embeddings)
        
        return AlignedTriplet(
            head=head,
            relation=relation,
            tail=tail,
            head_kg_emb=torch.tensor(head_emb),
            relation_kg_emb=torch.tensor(rel_emb),
            tail_kg_emb=torch.tensor(tail_emb),
            aligned_prefix_tokens=aligned_prefix.squeeze(0),  # Remove batch dim
            description=description
        )


class ReviewActionWithPrefixTokens:
    """
    Review action that properly uses prefix tokens for triplet verification
    """
    
    def __init__(
        self,
        llm_model_name: str = "meta-llama/Llama-3-8B-Instruct",
        kg_embedding_dim: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # Load base LLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.base_model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Initialize alignment module
        self.alignment_module = KGtoLLMAlignmentModule(
            kg_embedding_dim=kg_embedding_dim,
            llm_hidden_dim=self.base_model.config.hidden_size,
            llm_model_name=llm_model_name
        ).to(self.device)
        
        # Configure LoRA for fine-tuning
        lora_config = LoraConfig(
            r=32,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.base_model, lora_config)
        
    def create_input_with_prefix_tokens(
        self,
        aligned_triplet: AlignedTriplet,
        instruction: str
    ) -> Dict[str, torch.Tensor]:
        """
        Create LLM input with aligned prefix tokens prepended
        This is the key innovation - prefix tokens carry KG structural information
        """
        # Tokenize the instruction
        instruction_text = f"""### Instruction:
Given a triple from a knowledge graph. Each triple consists of a head entity, a relation, and a tail entity.
Please determine the correctness of the triple and response True or False.

### Description:
{aligned_triplet.description}

### Triple:
({aligned_triplet.head}, {aligned_triplet.relation}, {aligned_triplet.tail})

### Response:"""
        
        instruction_tokens = self.tokenizer(
            instruction_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Get instruction embeddings
        instruction_embeddings = self.base_model.get_input_embeddings()(
            instruction_tokens["input_ids"]
        )
        
        # Prepend aligned prefix tokens to instruction embeddings
        # Shape: prefix_tokens [3, hidden_dim], instruction [1, seq_len, hidden_dim]
        prefix_tokens = aligned_triplet.aligned_prefix_tokens.unsqueeze(0).to(self.device)
        
        # Concatenate prefix tokens with instruction embeddings
        combined_embeddings = torch.cat(
            [prefix_tokens, instruction_embeddings],
            dim=1  # Concatenate along sequence dimension
        )
        
        # Create attention mask for combined sequence
        prefix_mask = torch.ones(1, 3, device=self.device)
        instruction_mask = instruction_tokens["attention_mask"]
        combined_mask = torch.cat([prefix_mask, instruction_mask], dim=1)
        
        return {
            "inputs_embeds": combined_embeddings,
            "attention_mask": combined_mask
        }
    
    def verify_triplet(
        self,
        triplet: Tuple[str, str, str],
        kg_embeddings: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> bool:
        """
        Verify if a triplet is correct using prefix token augmented LLM
        """
        # Prepare aligned triplet with prefix tokens
        aligned_triplet = self.alignment_module.prepare_prefix_tokens(
            triplet=triplet,
            kg_embeddings=kg_embeddings,
            llm_model=self.base_model
        )
        
        # Create input with prefix tokens
        model_inputs = self.create_input_with_prefix_tokens(
            aligned_triplet=aligned_triplet,
            instruction=""  # Instruction is already in the method
        )
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=model_inputs["inputs_embeds"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=10,
                temperature=0.1,  # Low temperature for deterministic output
                do_sample=False
            )
        
        # Decode only the generated tokens (skip input)
        input_length = model_inputs["attention_mask"].sum().item()
        generated_tokens = outputs[0, input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Check if response indicates triplet is true
        return "true" in response.lower()
    
    def fine_tune_on_kg_completion(
        self,
        training_triplets: List[Tuple[Tuple[str, str, str], Tuple[np.ndarray, np.ndarray, np.ndarray], bool]],
        num_epochs: int = 3,
        learning_rate: float = 1e-4
    ):
        """
        Fine-tune the model on KG completion task using prefix tokens
        Training triplets format: [(triplet, kg_embeddings, is_correct), ...]
        """
        optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.alignment_module.parameters()),
            lr=learning_rate
        )
        
        self.model.train()
        self.alignment_module.train()
        
        for epoch in range(num_epochs):
            total_loss = 0
            
            for triplet, kg_embeddings, is_correct in training_triplets:
                # Prepare aligned triplet
                aligned_triplet = self.alignment_module.prepare_prefix_tokens(
                    triplet=triplet,
                    kg_embeddings=kg_embeddings,
                    llm_model=self.base_model
                )
                
                # Create input with prefix tokens
                model_inputs = self.create_input_with_prefix_tokens(
                    aligned_triplet=aligned_triplet,
                    instruction=""
                )
                
                # Prepare target (True or False)
                target_text = "True" if is_correct else "False"
                target_tokens = self.tokenizer(
                    target_text,
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    inputs_embeds=model_inputs["inputs_embeds"],
                    attention_mask=model_inputs["attention_mask"],
                    labels=target_tokens["input_ids"]
                )
                
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(training_triplets)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        self.alignment_module.eval()


# Example usage
def demonstrate_alignment():
    """
    Demonstrate the proper prefix token integration
    """
    # Initialize Review action with proper alignment
    review_action = ReviewActionWithPrefixTokens(
        llm_model_name="meta-llama/Llama-3-8B-Instruct",
        kg_embedding_dim=128
    )
    
    # Example triplet
    triplet = ("HSPA8", "interacts_with", "DHDDS")
    
    # Simulated TransE embeddings (would come from trained TransE model)
    kg_embeddings = (
        np.random.randn(128),  # HSPA8 embedding
        np.random.randn(128),  # interacts_with embedding
        np.random.randn(128)   # DHDDS embedding
    )
    
    # Generate description using dictionary
    description = review_action.alignment_module.generate_triplet_description(*triplet)
    print(f"Generated Description: {description}")
    
    # Verify triplet using prefix tokens
    is_valid = review_action.verify_triplet(triplet, kg_embeddings)
    print(f"Triplet verification result: {is_valid}")
    
    # Example training data for fine-tuning
    training_data = [
        (("Protein1", "interacts_with", "Protein2"), (np.random.randn(128), np.random.randn(128), np.random.randn(128)), True),
        (("Drug1", "contraindication", "Disease1"), (np.random.randn(128), np.random.randn(128), np.random.randn(128)), False),
        # Add more training examples...
    ]
    
    # Fine-tune model (would be done offline)
    # review_action.fine_tune_on_kg_completion(training_data, num_epochs=3)
    
    return review_action


if __name__ == "__main__":
    # Demonstrate the alignment process
    review_action = demonstrate_alignment()
