from __future__ import annotations

from re import T
from typing import Optional
from pydantic import BaseModel, Field

from environment.execution import (
    extract_format,
    verify_answer,
)

class Reward(BaseModel):
    """
    Reward for the curriculum and answer.
    """
    reward:str = Field(..., description="Reward for the curriculum and answer.")
    error:Optional[str] = Field(None, description="Error message if the reward cannot be computed.")
    feedback:Optional[str] = Field(None, description="Feedback for the curriculum and answer.")


class CurriculumReward(Reward):
    """
    Reward for the curriculum.
    """
    curriculum_generation: str = Field(..., description="The curriculum generation.")
    extracted_curriculum: Optional[str] = Field(None, description="The extracted curriculum.")


class AnswerReward(Reward):
    """
    Reward for the answer.
    """
    answer_generation: str = Field(..., description="The answer generation.")
    extracted_answer: Optional[str] = Field(None, description="The extracted answer.")


def reward_curriculum(
    curriculum_generation: str, 
    answer_reward:Optional[AnswerReward] = None, 
    check_format_only: bool = False, 
    format_regex:str = r"```python\n(.*?)\n```",
    use_last_match: bool = True,
    variable_name: str = "x",
    ) -> Reward:
    """
    Reward the curriculum based on the answer.

    Args:
        curriculum: The curriculum to reward.
        answer_group_reward: The answer group reward.
        check_format_only: Whether to only check the format of the answer or to execute the answer.
        
    Returns:
        Reward: The reward for the curriculum and answer.
    """
    ## check the format 
    extracted_curriculum = extract_format(
        text=curriculum_generation, 
        regex_format=format_regex, 
        use_last_match=use_last_match
    )
    if not extracted_curriculum:
        return CurriculumReward(
            curriculum_generation=curriculum_generation,
            extracted_curriculum=extracted_curriculum,
            reward=0.0,
            error=f"No curriculum satisfying the format {format_regex} extracted from the curriculum generation.",
            feedback=f"Please ensure the curriculum satisfies the format {format_regex}.",
        )

    # run the curriculum with a dummy answer
    verification = verify_answer(
        curriculum=extracted_curriculum,
        variable_assignment=1,
        variable_name="x"
    )
    if verification.error is not None:
        return CurriculumReward(
            curriculum_generation=curriculum_generation,
            extracted_curriculum=extracted_curriculum,
            reward=0.0,
            error=verification.error,
            feedback=f"Please ensure the curriculum satisfies the format {format_regex}.",
        )

    
    # 
    

def reward_answer(curriculum: str, answer: str) -> Reward:
    """
    Reward the answer based on the curriculum.
    """
    ... 