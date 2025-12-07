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
        return f"#Name:{self.name}\n#Description: {self.description}\n#Format: {self.format}"


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
    ## Core task objective:
    Any problem belongs to the task is defined by a set of executable boolean constraints about a single variable "x".
    To solve a problem designed, a student must write Python code to derive a particular value of "x", which meets all constraints imposed.

    # Properties of the task:
    * Any problem designed must be solveable.
    * The student must be able to exercise sophisticated logical reasoning, math and programming skills to solve problems that you've designed.
    """
    ).strip()

    format: str = dedent(
    """
    For each problem created belonging to the task, put your final designed problem inside the <answer> tag. DO NOT provide the solution/any hint. 
    <answer>
    Write Python code to derive a particular value of "x" which meets all constraints imposed:
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
