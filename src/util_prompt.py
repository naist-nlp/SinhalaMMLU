def generate_instruction_prompt(level, subject, subject_original, question, choices, language="sinhala", few_shot_examples=None, intro_type=True):
    """
    Generates the instruction prompt with optional few-shot examples.

    Args:
        level (str): The difficulty level (easy, medium, hard).
        subject (str): The subject name.
        question (str): The actual question.
        choices (list): List of answer choices.
        language (str): "sinhala" or "english" (default: sinhala).
        few_shot_examples (list, optional): Few-shot examples for reference.
        intro_type (bool): Whether to include the subject introduction in the prompt (default: True).

    Returns:
        str: The optimized instruction prompt.
    """

    # Sinhala & English introductions
    sinhala_intro = f"මෙය {subject_original} විෂයයට අදාළ බහුවරණ ප්‍රශ්නයකි.\n" if intro_type else ""
    english_intro = f"This is a multiple choice question related to the subject {subject}. Given in Sinhala Language.\n" if intro_type else ""

    # Instructions based on level
    sinhala_instructions = {
        "easy": "පහත ප්‍රශ්නයට 1, 2, 3, 4 යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරන්න.\n\n",
        "medium": "පහත ප්‍රශ්නයට 1, 2, 3, 4  යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරාගන්න.\n\n",
        "hard": "පහත ප්‍රශ්නයට 1, 2, 3, 4, 5 යන පිළිතුරුවලින් නිවැරදි හෝ ඉතාමත් ගැළපෙන පිළිතුර තෝරාගන්න.\n\n"
    }

    english_instructions = {
        "easy": "Choose the correct or most appropriate answer from 1, 2, 3, 4 for the question below.\n\n",
        "medium": "Choose the correct or most appropriate answer from 1, 2, 3, 4 for the question below.\n\n",
        "hard": "Select the correct answer from 1, 2, 3, 4, 5 for the given question.\n\n"
    }

    if level.lower() not in sinhala_instructions:
        raise ValueError(f"Unsupported difficulty level: {level}")

    # Select language-based instruction
    instruction_text = sinhala_instructions if language.lower() == "sinhala" else english_instructions

    # Format few-shot examples properly with Sinhala/English intro & difficulty-based instructions!
    few_shot_prompt = ""
    if few_shot_examples:
        few_shot_prompt = "\n".join(
            (sinhala_intro if intro_type and language.lower() == "sinhala" else "")  # Add intro only if intro_type=True
            + (english_intro if intro_type and language.lower() == "english" else "")  # Add intro only if intro_type=True
            + (sinhala_instructions[level.lower()] if language.lower() == "sinhala" else english_instructions[level.lower()])  # Add difficulty-based instructions
            + f"{'ප්‍රශ්නය' if language.lower() == 'sinhala' else 'Question'}: {ex['question']}\n"
            + "\n".join([f"{j+1}. {choice.strip()}" for j, choice in enumerate(ex['choices'])])
            + f"\n{'පිළිතුර' if language.lower() == 'sinhala' else 'Answer'}: {ex['answer']}\n"
            for i, ex in enumerate(few_shot_examples)
        ) + "\n\n"


    # Format few-shot examples properly with Sinhala/English intro & difficulty-based instructions!
    # few_shot_prompt = ""
    # if few_shot_examples:
    #     few_shot_prompt = "\n".join(
    #         (sinhala_intro if language.lower() == "sinhala" else english_intro)  # Include Sinhala/English intro
    #         + (sinhala_instructions[level.lower()] if language.lower() == "sinhala" else english_instructions[level.lower()])  # Add difficulty-based instructions
    #         + f"{'ප්‍රශ්නය' if language.lower() == 'sinhala' else 'Question'}: {ex['question']}\n"
    #         + "\n".join([f"{j+1}. {choice.strip()}" for j, choice in enumerate(ex['choices'])])
    #         + f"\n{'පිළිතුර' if language.lower() == 'sinhala' else 'Answer'}: {ex['answer']}\n"
    #         for i, ex in enumerate(few_shot_examples)
    #     ) + "\n\n"

    # Format the final instruction
    formatted_choices = "\n".join([f"{i+1}. {choice.strip()}" for i, choice in enumerate(choices)])


    if language.lower() == "sinhala":
        return (
            few_shot_prompt  # Attach properly formatted few-shot examples (if any)
            + (sinhala_intro if intro_type and language.lower() == "sinhala" else "")  # Fix the condition
            + instruction_text[level.lower()]
            + f"ප්‍රශ්නය: {question}\n{formatted_choices}\nපිළිතුර:"
        )
    else:
        return (
            few_shot_prompt
            + (english_intro if intro_type and language.lower() == "english" else "")  # Fix the condition
            + instruction_text[level.lower()]
            + f"Question: {question}\n{formatted_choices}\nAnswer:"
        )
    
