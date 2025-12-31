from __future__ import annotations

import asyncio
from functools import partial
from typing import Optional, List, Callable
from pydantic import BaseModel, Field
from textwrap import dedent
from environment.execution import (
    verify,
)
from environment.format import (
    TEACHER_PATTERN,
    STUDENT_PATTERN,
    extract_format,
)

class Reward(BaseModel):
    """
    Reward for the curriculum and answer.
    """
    reward: Optional[float] = Field(
        None, description="Reward for the curriculum and answer."
    )
    error: Optional[str] = Field(
        None, description="Error message if the reward cannot be computed."
    )
    feedback: Optional[str] = Field(
        None, description="Feedback for the curriculum and answer."
    )

    @property
    def passed(self) -> bool:
        return self.reward >= 1.0

    @property
    def format_passed(self) -> bool: ...

    @property
    def only_format_passed(self) -> bool: ...


class CurriculumReward(Reward):
    """
    Reward for the curriculum.
    """

    curriculum_generation: str = Field(..., description="The curriculum generation.")
    extracted_curriculum: Optional[str] = Field(
        None, description="The extracted curriculum."
    )
    format_only: bool = Field(
        False,
        description="Whether to only check the format of the curriculum or to execute the curriculum.",
    )

    @property
    def format_passed(self) -> bool:
        if self.format_only:
            return self.reward >= 1.0

    @property
    def only_format_passed(self) -> bool:
        if self.format_only:
            return self.reward >= 1.0
        else:
            return False

    def __str__(self) -> str:
        return dedent(
            f"""
            ===============================
            Curriculum Reward:
            - Format Only: {self.format_only}
            - Reward: {self.reward}
            - Error: {self.error}
            - Feedback: {self.feedback}
            ## Extracted Curriculum
            {self.extracted_curriculum.strip() if self.extracted_curriculum else "None"}
            ===============================
            """
        ).strip()


class AnswerReward(Reward):
    """
    Reward for the answer.
    """

    answer_generation: str = Field(..., description="The answer generation.")
    extracted_answer: Optional[str] = Field(None, description="The extracted answer.")


def reward_curriculum(
    curriculum_generation: str,
    format_regex: str,
    answer_group_rewards: Optional[list[AnswerReward]] = None,
    check_format_only: bool = False,
    variable_name: str = "x",
    check_diversity: bool = False,
) -> CurriculumReward:
    """
    Reward the curriculum based on the answer.

    Args:
        curriculum: The curriculum to reward.
        answer_group_reward: The answer group reward.
        check_format_only: Whether to only check the format of the answer or to execute the answer.
        check_diversity: Whether to check the diversity of the answer.
        format_regex: The regex to extract the curriculum.
        use_last_match: Whether to use the last match or the first match.
        variable_name: The variable name to use for the answer.
    Returns:
        Reward: The reward for the curriculum and answer.
    """
    ## check the format
    extracted_curriculum = extract_format(
        text=curriculum_generation,
        regex_format=format_regex,
    )
    if not extracted_curriculum:
        return CurriculumReward(
            extracted_curriculum=extracted_curriculum,
            reward=0.0,
            curriculum_generation=curriculum_generation,
            error=f"No curriculum satisfying the format {format_regex} extracted from the curriculum generation.",
            feedback=f"Please ensure the curriculum satisfies the format {format_regex}.",
            format_only=check_format_only,
        )

    # run the curriculum with a dummy answer
    runtime_verification = verify(code=extracted_curriculum, variable_assignment=1)
    if runtime_verification.error is not None:
        return CurriculumReward(
            curriculum_generation=curriculum_generation,
            extracted_curriculum=extracted_curriculum,
            reward=0.0,
            error=runtime_verification.error,
            feedback=f"Curriculum satisfied the format, however the python code is not runnable.",
            format_only=check_format_only,
        )

    if check_format_only:
        return CurriculumReward(
            curriculum_generation=curriculum_generation,
            extracted_curriculum=extracted_curriculum,
            reward=1.0,
            error=None,
            feedback=f"The curriculum satisfies the format {format_regex}.",
            format_only=check_format_only,
        )

    if answer_group_rewards is None:
        raise ValueError(
            "answer_group_rewards is required when check_format_only is False"
        )

    solved_rate: float = [
        answer_reward.passed for answer_reward in answer_group_rewards
    ].count(True) / len(answer_group_rewards)
    uncertainty_reward: float = solved_rate * (1 - solved_rate)

    # diversity reward
    # TODO: implement diversity reward
    diversity_reward: float = 0.0

    return CurriculumReward(
        curriculum_generation=curriculum_generation,
        extracted_curriculum=extracted_curriculum,
        reward=uncertainty_reward + diversity_reward,
        format_only=False,
    )

def reward_answer(
    extracted_curriculum: str,
    answer_generation: str,
    format_regex: str,
    variable_name: str = "x",
    check_diversity: bool = False,
) -> AnswerReward:
    """
    Reward the answer based on the curriculum.
    
    For student answers in \\boxed{} format, the extracted value is evaluated
    directly as a Python expression (e.g., "26" -> 26, "-5" -> -5, "3.14" -> 3.14).
    """
    # check the format
    extracted_answer = extract_format(
        text=answer_generation, 
        regex_format=format_regex
    )
    if not extracted_answer:
        return AnswerReward(
            answer_generation=answer_generation,
            extracted_answer=extracted_answer,
            reward=0.0,
            error=f"No answer satisfying the format extracted from the answer generation.",
            feedback=f"Please ensure the answer is in the correct format (e.g., \\boxed{{value}}).",
        )

    # For \boxed{} format, evaluate the extracted value directly
    # The extracted_answer is just the value inside \boxed{}, e.g., "26"
    try:
        answer_value = eval(extracted_answer)
    except Exception as e:
        return AnswerReward(
            answer_generation=answer_generation,
            extracted_answer=extracted_answer,
            reward=0.0,
            error=f"Could not evaluate answer '{extracted_answer}': {e}",
            feedback=f"The answer must be a valid numeric expression.",
        )

    verification_result = verify(
        code=extracted_curriculum, variable_assignment=answer_value
    )
    if verification_result.error is not None:
        return AnswerReward(
            answer_generation=answer_generation,
            extracted_answer=extracted_answer,
            reward=0.0,
            error=verification_result.error,
            feedback=f"The answer format is correct, but verification failed.",
        )

    return AnswerReward(
        answer_generation=answer_generation,
        extracted_answer=extracted_answer,
        reward=1.0 if verification_result.correct else 0.0,
    )

async def _reward_answer_async(
    extracted_curriculum: str,
    answer_generation: str,
    format_regex: str,
    use_last_match: bool,
    variable_name: str,
    check_diversity: bool,
) -> AnswerReward:
    """Async wrapper around reward_answer without spawning threads or processes."""
    return reward_answer(
        extracted_curriculum=extracted_curriculum,
        answer_generation=answer_generation,
        format_regex=format_regex,
        use_last_match=use_last_match,
        variable_name=variable_name,
        check_diversity=check_diversity,
    )


async def reward_answer_group(
    extracted_curriculum: str,
    answer_generations: List[str],
    format_regex: str = STUDENT_PATTERN,
    use_last_match: bool = True,
    variable_name: str = "x",
    check_diversity: bool = False,
) -> List[AnswerReward]:
    """
    Reward a group of answers based on the curriculum using asyncio.TaskGroup.
    No threads or executors are used; the synchronous work runs directly in each task.
    """
    tasks: list[asyncio.Task[AnswerReward]] = []
    async with asyncio.TaskGroup() as tg:
        for answer_generation in answer_generations:
            tasks.append(
                tg.create_task(
                    _reward_answer_async(
                        extracted_curriculum=extracted_curriculum,
                        answer_generation=answer_generation,
                        format_regex=format_regex,
                        use_last_match=use_last_match,
                        variable_name=variable_name,
                        check_diversity=check_diversity,
                    )
                )
            )

    return [task.result() for task in tasks]


# Partial version that only checks format (does not require answer_group_rewards)
function_reward_curriculum_format_only: Callable[[str], CurriculumReward] = partial(
    reward_curriculum, format_regex=TEACHER_PATTERN, check_format_only=True
)

function_reward_answer: Callable[[tuple[str,str]], AnswerReward] = partial(
    reward_answer, format_regex=STUDENT_PATTERN,  variable_name="x", check_diversity=False
)