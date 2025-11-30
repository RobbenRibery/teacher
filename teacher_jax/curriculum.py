from  __future__ import annotations
from pydantic import BaseModel, Field
from textwrap import dedent

from tunix.models.qwen3 import model as qwen3_lib

CURRICULM_MAKER_SYSTEM_PROMPT = dedent("""
You are an expert curriculum designer. Generate diverse and challenging instructions based on the instruction provided.
""")

class Instruction(BaseModel):
    """
    Represents a single instruction generated for the curriculum.
    """
    text: str = Field(..., description="The content of the instruction.")

class Curriculums(BaseModel):
    """
    A collection of instructions forming a curriculum.
    """
    instructions: list[Instruction] = Field(..., description="List of generated instructions.")

def generate_curriculum(curriculum_meta_instruction: str, model_name: str, **kwargs) -> Curriculums:
    """
    Generates a curriculum of instructions based on a high-level meta-instruction.

    This function uses a Pydantic AI agent to generate a structured list of instructions.
    It takes a meta-instruction (e.g., "math problems for grade 5") and returns a
    `Curriculums` object containing specific `Instruction` objects.

    Args:
        curriculum_meta_instruction (str): The high-level description or goal for the curriculum.

    Returns:
        Curriculums: A structured object containing the generated instructions.
    """
    
    model = qwen3_lib.create_model
