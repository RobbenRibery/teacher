from __future__ import annotations

from pydantic import BaseModel, Field
from textwrap import dedent


class Task(BaseModel):
    """
    A task
    """

    name: str = Field(..., description="The name of the task.")
    description: str = Field(..., description="The description of the task.")
    format: str = Field(
        ...,
        description="The format of the task, must contain a 'seed_task' to be filled.",
    )

    def render(self) -> str:
        return f"## Name:\n{self.name}\n\n## Description:\n{self.description}\n\n## Format:\n{self.format}"


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
    ### Objective:
    Any Boolean Constraints Satisfaction problem is defined by a set of executable boolean constraints about a variable "x", which can be any real number. 
    Each constraint can be implemented as a Python lambda function that takes in "x" and output True/False.
    To solve such a problem, a student must reason step by step to derive "x" that meets all constraints imposed. 
    
    ### Properties of the curriculum
    * At least 1 solution to x must exist, though problems can have multiple solutions.
    * The problem must be non-trival, such that the student must be able to exercise reasoning skills while solving the problem.
    * Only a single variable x is allowed.
    """
    ).strip()

    format: str = dedent(
        """
    ### Guideline
    - Put your final designed problem inside exactly 1 ```python\n...\n``` block and don't include solutions or hints.
    - Copy the commented line "# Reason step by step and derive the final value of x." inside the python code block.
    - End right after the python code block.

    ### Example final section of the response
    ```python
    # Reason step by step and derive the final value of x.
    constraints = [
        lambda x: ...,
        lambda x: ...,
        ...,
    ]
    ```
    """
    ).strip()


TASKS = {"boolean_constraints_satisfaction": BooleanConstraintsSatisfaction}
