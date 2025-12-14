from __future__ import annotations

from pydantic import BaseModel, Field
from textwrap import dedent

class Task(BaseModel):
    """
    A task
    """
    name: str = Field(
        ...,
        description="The name of the task."
    )
    description: str = Field(
        ...,
        description="The description of the task."
    )
    format: str = Field(
        ...,
        description="The format of the task, must contain a 'seed_task' to be filled."
    )

    def render(self) -> str:
        return f"#Name:{self.name}\n\n#Description:\n{self.description}\n\n#Format:\n{self.format}"


class BooleanConstraintsSatisfaction(Task):
    """
    A boolean constraint satisfaction task.
    """
    name: str = dedent(
    """
    Boolean Constraints Satisfaction
    """
    ).strip()
    
    description: str = dedent(
    """
    ## Objective:
    Each problem from the curriculum is defined by a set of executable boolean constraints about a single variable "x".
    To solve a problem from the curriculum, a student must write Python code to derive/search/find/compute a particular value of "x" that meets all constraints imposed.

    # Properties of the curriculum
    * Any problems from the curriculum must be solvable.
    * The student must be able to exercise sophisticated logical reasoning, math, and programming skills to solve the problems you design.
    """
    ).strip()

    format: str = dedent(
    """
    Put your final designed problem inside the <answer> tag and don't include solution or hints inside the final answer.
    Use the following format for think and answer. 
    <think>
    ...
    </think>
    <answer>
    ```python
    # Write Python code to derive a particular value of "x" that meets all constraints imposed:
    constraints = [
        lambda x: ...,
        lambda x: ...,
        ...
    ]
    ```
    </answer> (terminate here and no further rounds)
    """
    ).strip()

TASKS = {
    "boolean_constraints_satisfaction": BooleanConstraintsSatisfaction
}
