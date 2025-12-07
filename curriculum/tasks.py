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
    The task is defined by a set of executable boolean constraints.
    To solve the task, a solver must reason step by step and then write a python function, which determines the numerical value of a variable "x". 
    The output value of such function must meet the constraints as imposed by the task.
    The task can have multiple solutions.

    ## Ultimate objective from solving the task:
    The solver must be able to exercise logical reasoning, math and programming skills from solving instances of the task.
    """
    ).strip()

    format: str = dedent(
    """
    Final output for the task creator (only applies for content inside the <answer> tag), must contain a brief description of the function and executable boolean constraints.
    <answer>
    brief description here... (i.e. Write a python function that outputs the numerical value of a variable "x" which meets the following constraints: ...).
    ```python
    set of boolean constraints... # example: (x > 0) and (x < 10 or x == 5)
    ```
    </answer>
    """
    ).strip()


TASKS = {
    "boolean_constraints_satisfaction": BooleanConstraintsSatisfaction
}
