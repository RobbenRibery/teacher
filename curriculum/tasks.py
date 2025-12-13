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
    ## Core Curriculum Objective:
    Each instance in this curriculum is defined by a set of executable boolean constraints about a single variable "x".
    To solve an instance, a student must write Python code to derive a particular value of "x" that meets all constraints imposed.

    # Properties of the task:
    * Any instance designed must be solvable.
    * The student must be able to exercise sophisticated logical reasoning, math, and programming skills to solve the instances you design.
    """
    ).strip()

    format: str = dedent(
    """
    For each instance you create, put your final designed instance inside the <answer> tag. DO NOT provide the solution or any hints.
    Example format for the final answer:
    <answer>
    Write Python code to derive a particular value of "x" that meets all constraints imposed:
    ```python
    constraints = [
        lambda x: ...,
        lambda x: ...,
        ...
    ]
    ```
    </answer>
    """
    ).strip()

TASKS = {
    "boolean_constraints_satisfaction": BooleanConstraintsSatisfaction
}
