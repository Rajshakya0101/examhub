"""
Question Quality Validator for ExamHub

Ensures all AI-generated questions meet SSC exam standards.
Rejects fake, unrealistic, or low-quality questions. 
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("question_validator")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class QuestionValidator:
    """Validates SSC-style questions for authenticity and quality."""
    
    def __init__(self):
        # Track question hashes to detect duplicates
        self.seen_questions = set()
        
        # Common fake/placeholder patterns to reject
        self.fake_patterns = [
            r"lorem ipsum",
            r"example question",
            r"sample answer",
            r"placeholder",
            r"<fill.*?>",
            r"TODO",
            r"XXX",
            r"\[blank\]",
        ]
        
        # SSC-appropriate number ranges (to catch absurd values)
        self.reasonable_ranges = {
            "percentage": (0, 500),  # up to 500% for some scenarios
            "money": (1, 1000000),   # Rs. 1 to 10 lakhs typically
            "age": (1, 120),
            "speed": (1, 500),       # km/h
            "time_hours": (0.1, 100),
            "distance": (0.1, 10000), # km
            "weight": (0.1, 10000),  # kg
          }
    
    def validate_question(self, mcq: Dict[str, Any], subject: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a single MCQ for quality and authenticity.
        
        Returns:
            (is_valid, rejection_reason)
        """
        try:
            # 1. Basic structure validation
            is_valid, reason = self._validate_structure(mcq)
            if not is_valid:
                return False, reason
            
            # 2. Check for fake/placeholder content
            is_valid, reason = self._check_fake_content(mcq)
            if not is_valid:
                return False, reason
            
            # 3.  Validate options quality
            is_valid, reason = self._validate_options(mcq)
            if not is_valid:
                return False, reason
            
            # 4. Check numerical reasonability (for quant questions)
            if "quant" in subject.lower() or "math" in subject.lower():
                is_valid, reason = self._validate_numerical_reasonability(mcq)
                if not is_valid:
                    return False, reason
            
            # 5.  Validate explanation quality
            is_valid, reason = self._validate_explanation(mcq)
            if not is_valid:
                return False, reason
            
            # 6. Check for duplicates
            is_valid, reason = self._check_duplicate(mcq)
            if not is_valid:
                return False, reason
            
            # All checks passed
            return True, None
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False, f"Validation exception: {e}"
    
    def _validate_structure(self, mcq: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check basic structure requirements."""
        required_fields = [
            "questionText", "optionA", "optionB", "optionC", "optionD",
            "correctOption", "explanation"
        ]
        
        for field in required_fields:
            if field not in mcq or not mcq[field]:
                return False, f"Missing or empty field: {field}"
        
        # Question must be substantial (not too short)
        if len(mcq["questionText"]) < 20:
            return False, "Question text too short (likely incomplete)"
        
        # Explanation must be substantial
        if len(mcq["explanation"]) < 15:
            return False, "Explanation too short (not detailed enough)"
        
        # Correct option must be valid
        if mcq["correctOption"] not in ["A", "B", "C", "D"]:
            return False, f"Invalid correct option: {mcq['correctOption']}"
        
        return True, None
    
    def _check_fake_content(self, mcq: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Detect fake/placeholder content."""
        all_text = " ".join([
            mcq["questionText"],
            mcq["optionA"],
            mcq["optionB"],
            mcq["optionC"],
            mcq["optionD"],
            mcq["explanation"],
        ]).lower()
        
        for pattern in self.fake_patterns:
            if re.search(pattern, all_text, re. IGNORECASE):
                return False, f"Contains placeholder/fake content: {pattern}"
        
        # Check for suspiciously generic language
        generic_phrases = [
            "insert value here",
            "calculate this",
            "solve for x",  # without actual problem
            "find the answer",
            "none of these",  # suspicious if used as correct answer frequently
        ]
        
        for phrase in generic_phrases:
            if phrase in all_text and len(mcq["questionText"]) < 50:
                return False, f"Too generic/vague: contains '{phrase}'"
        
        return True, None
    
    def _validate_options(self, mcq: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate answer options quality."""
        options = [
            mcq["optionA"],
            mcq["optionB"],
            mcq["optionC"],
            mcq["optionD"],
        ]
        
        # 1. All options must be non-empty
        for i, opt in enumerate(options, 1):
            if not opt or len(opt. strip()) == 0:
                return False, f"Option {chr(64+i)} is empty"
        
        # 2. Options should not be identical
        unique_options = set(opt.strip(). lower() for opt in options)
        if len(unique_options) < 4:
            return False, "Options are not distinct (duplicates found)"
        
        # 3. Options should be similar in format/length (not extreme differences)
        lengths = [len(opt) for opt in options]
        max_len = max(lengths)
        min_len = min(lengths)
        
        # If one option is 10x longer than others, suspicious
        if max_len > min_len * 10 and min_len > 0:
            return False, "Options have suspiciously different lengths"
        
        # 4. For numerical answers, check if they're reasonable
        try:
            numerical_options = []
            for opt in options:
                # Extract number from string like "25%", "Rs. 100", "20 km"
                numbers = re.findall(r'-?\d+\.?\d*', opt)
                if numbers:
                    numerical_options.append(float(numbers[0]))
            
            if len(numerical_options) == 4:
                # All options are numerical - check if they're distinct enough
                numerical_options.sort()
                # Check if options are too close together (likely fake)
                for i in range(len(numerical_options) - 1):
                    if numerical_options[i] == numerical_options[i+1]:
                        return False, "Numerical options contain duplicates"
        except:
            pass  # Non-numerical options, skip this check
        
        return True, None
    
    def _validate_numerical_reasonability(self, mcq: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if numbers in the question are reasonable for SSC exams."""
        text = mcq["questionText"] + " " + " ".join([
            mcq["optionA"], mcq["optionB"], mcq["optionC"], mcq["optionD"]
        ])
        
        # Extract all numbers from the question
        numbers = re.findall(r'\d+\.?\d*', text)
        if not numbers:
            return True, None  # No numbers to validate
        
        numbers = [float(n) for n in numbers]
        
        # Check for absurdly large or small numbers
        for num in numbers:
            # Numbers should typically be manageable for mental calculation
            if num > 1000000:  # 10 lakhs is usually the upper limit
                return False, f"Unrealistic number: {num} (too large for SSC pattern)"
            
            # Very precise decimals are suspicious (e.g., 3.14159265)
            if '.' in str(num):
                decimal_places = len(str(num).split('.')[1])
                if decimal_places > 4:
                    return False, f"Unrealistic precision: {num} (SSC uses simpler numbers)"
        
        # Check for suspiciously perfect numbers (might indicate fake data)
        perfect_numbers = [1000000, 999999, 123456, 111111, 222222]
        for num in numbers:
            if num in perfect_numbers:
                logger.warning(f"Suspicious perfect number found: {num}")
        
        return True, None
    
    def _validate_explanation(self, mcq: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate explanation quality."""
        explanation = mcq["explanation"]. strip()
        
        # 1. Must not be a generic statement
        generic_explanations = [
            "explanation not provided",
            "solve using standard method",
            "apply the formula",
            "calculate accordingly",
            "use basic math",
        ]
        
        explanation_lower = explanation.lower()
        for generic in generic_explanations:
            if generic in explanation_lower and len(explanation) < 50:
                return False, f"Explanation too generic: '{explanation[:50]}... '"
        
        # 2. For quant questions, should contain some calculation/formula
        if "quant" in mcq. get("subject", "").lower():
            has_calculation = any([
                char in explanation for char in ['=', '+', '-', '*', '/', '%', '×', '÷']
            ]) or any([
                keyword in explanation_lower 
                for keyword in ['formula', 'calculate', 'multiply', 'divide', 'add', 'subtract']
            ])
            
            if not has_calculation and len(explanation) < 100:
                return False, "Quantitative question lacks proper calculation in explanation"
        
        # 3. Should reference the correct option or show why it's correct
        correct_option = mcq["correctOption"]
        correct_value = mcq[f"option{correct_option}"]
        
        # Extract numbers from explanation and correct answer
        explanation_numbers = set(re.findall(r'\d+\.?\d*', explanation))
        answer_numbers = set(re.findall(r'\d+\.?\d*', correct_value))
        
        # If the answer contains numbers, explanation should mention them
        if answer_numbers and not explanation_numbers. intersection(answer_numbers):
            logger.warning(f"Explanation doesn't clearly show how to arrive at answer: {correct_value}")
        
        return True, None
    
    def _check_duplicate(self, mcq: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check for duplicate questions."""
        # Create a hash of the question (ignore minor variations)
        question_normalized = re.sub(r'\s+', ' ', mcq["questionText"].lower(). strip())
        question_hash = hash(question_normalized)
        
        if question_hash in self. seen_questions:
            return False, "Duplicate question detected"
        
        self.seen_questions.add(question_hash)
        return True, None
    
    def validate_batch(self, mcqs: List[Dict[str, Any]], subject: str) -> List[Dict[str, Any]]:
        """
        Validate a batch of MCQs and return only the valid ones.
        
        Args:
            mcqs: List of MCQ dictionaries
            subject: Subject name for context-specific validation
        
        Returns:
            List of valid MCQs (rejects may be logged)
        """
        valid_mcqs = []
        rejected_count = 0
        
        for idx, mcq in enumerate(mcqs, 1):
            is_valid, reason = self.validate_question(mcq, subject)
            
            if is_valid:
                valid_mcqs.append(mcq)
                logger.info(f"✓ Question {idx} passed validation")
            else:
                rejected_count += 1
                logger.warning(f"✗ Question {idx} REJECTED: {reason}")
                logger.debug(f"   Question text: {mcq. get('questionText', '')[:100]}...")
        
        logger.info(f"Validation complete: {len(valid_mcqs)} valid, {rejected_count} rejected")
        
        return valid_mcqs
    
    def reset_duplicate_tracker(self):
        """Reset the duplicate detection tracker (call between tests)."""
        self.seen_questions.clear()