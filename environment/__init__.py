from environment.format import (
    BASE_PATTERN,
    ANSWER_FORMAT_REGEX,
    TEACHER_PATTERN,
    STUDENT_PATTERN,
    CODE_BLOCK_REGEX,
    extract_format,
    extract_teacher_code,
    extract_student_code,
)

# Legacy aliases for backward compatibility
TEACHER_FORMAT_REGEX = TEACHER_PATTERN
STUDENT_FORMAT_REGEX = STUDENT_PATTERN


