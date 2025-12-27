You are a professor exceptionally skilled at crafting problem with high learning potential for students.

# Your Task
Design exactly 1 problem from the below curriculum.

# Curriculum Description
## Name:
Boolean Constraints Satisfaction

## Description:
### Objective:
Any Boolean Constraints Satisfaction problem is defined by a set of executable boolean constraints about a variable "x", which can be any real number. 
Each constraint can be implemented as a Python lambda function that takes in "x" and output True/False.
To solve such a problem, a student must reason step by step to derive "x" that meets all constraints imposed. 

### Properties of the curriculum
* At least 1 solution to x must exist, though problems can have multiple solutions.
* The student must be able to exercise a set of general skills (i.e. math, programming or logic) while solving the problem.
* Only a single variable x is allowed.

## Format:
### Guideline
- Put your final designed problem inside the <answer> tag and don't include solutions or hints.
- Your response MUST include exactly one <answer>...</answer> block.
- Any <think>...</think> content should appear BEFORE <answer>.
- Do not include any text after </answer>.

### Example
<answer>
```python
# Reason step by step and derive a solution of x.
constraints = [
    lambda x: ...,
    lambda x: ...,
    ...,
]
```
</answer>