import traceback
import math
import random
import itertools
import collections
import signal
from contextlib import contextmanager
from typing import Any, Optional, Callable
from pydantic import BaseModel, Field
from environment.format import extract_format

try:
    import numpy as np
except ImportError:
    np = None


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    def signal_handler(signum, frame):
        raise TimeoutException("Execution timed out.")

    try:
        original_handler = signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, original_handler)
    except ValueError:
        # Not in main thread, skip timeout
        yield


def get_exec_globals():
    """
    Returns a dictionary of safe/common globals to be available in exec.
    """
    g = {
        "math": math,
        "random": random,
        "itertools": itertools,
        "collections": collections,
    }
    if np:
        g["np"] = np
        g["numpy"] = np
    return g


class VerificationResult(BaseModel):
    """
    Model for verification results.
    """

    correct: bool = Field(
        ..., description="Whether the code is correct according to the constraints."
    )
    error: Optional[str] = Field(None, description="Traceback if the execution failed.")
    feedback: Optional[str] = Field(
        None, description="Feedback about constraint satisfaction."
    )


class ExecutionResult(BaseModel):
    """
    Model for execution results.
    """

    value: Any = Field(..., description="The value of the variable after execution.")
    error: Optional[str] = Field(None, description="Traceback if the execution failed.")


def verify(code: str, variable_assignment: Any) -> VerificationResult:
    """
    Validate that the curriculum is runnable by assigning variable name to a particular value.
    The curriculum should define a list of constraints (lambdas), e.g.:
    constraints = [lambda x: x > 0, lambda x: x < 10]

    Args:
        code: Python code string defining a 'constraints' list of callables.
        variable_assignment: The value to test against the curriculum.

    Returns:
        VerificationResult object with correct=True/False and error details if any.
    """
    try:
        local_scope = {}

        # Execute the curriculum code to define constraints
        with time_limit(30):
            exec(code, get_exec_globals(), local_scope)

        constraints = local_scope.get("constraints")

        if not isinstance(constraints, list):
            return VerificationResult(
                correct=False, error="Curriculum must define a 'constraints' list."
            )

        # Check that all constraints use 'x' as the variable name
        for constraint in constraints:
            if hasattr(constraint, "__code__"):
                if constraint.__code__.co_argcount != 1:
                    return VerificationResult(
                        correct=False,
                        error="Each constraint must take exactly one argument.",
                    )
                if constraint.__code__.co_varnames[0] != "x":
                    return VerificationResult(
                        correct=False,
                        error=f"Constraint argument must be named 'x', found '{constraint.__code__.co_varnames[0]}'.",
                    )

        satisfied_constraints: list[Callable] = []
        for _, constraint in enumerate(constraints):
            try:
                with time_limit(30):
                    if not constraint(variable_assignment):
                        return VerificationResult(
                            correct=False,
                            error=None,
                            feedback=f"Constraint {constraint} is not satisfied.",
                        )
                    else:
                        satisfied_constraints.append(constraint)
            except TimeoutException:
                return VerificationResult(
                    correct=False,
                    error="Execution timed out during constraint checking.",
                )
            except Exception:
                return VerificationResult(correct=False, error=traceback.format_exc())

        return VerificationResult(
            correct=True, error=None, feedback=f"All constraints are satisfied."
        )

    except TimeoutException:
        return VerificationResult(
            correct=False, error="Execution timed out during curriculum definition."
        )
    except Exception:
        return VerificationResult(correct=False, error=traceback.format_exc())


def execute_answer(extracted_answer: str, variable_name: str = "x") -> ExecutionResult:
    """
    Execute the extracted answer code to get the variable value.

    Args:
        extracted_answer: Python code to execute.
        variable_name: The name of the variable to extract after execution.

    Returns:
        ExecutionResult object with value and error details if any.
    """
    try:
        local_scope = {}

        # Execute the code
        with time_limit(30):
            exec(extracted_answer, get_exec_globals(), local_scope)

        # Return the variable value if it exists
        if variable_name in local_scope:
            return ExecutionResult(value=local_scope[variable_name], error=None)

        return ExecutionResult(
            value=None,
            error=f"Variable '{variable_name}' not found in the execution scope.",
        )
    except TimeoutException:
        return ExecutionResult(value=None, error="Execution timed out.")
    except Exception:
        return ExecutionResult(value=None, error=traceback.format_exc())
