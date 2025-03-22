import re
import json
import numpy as np
from openai import OpenAI
from pydantic import BaseModel

def describe_process(node, is_root=True):
    """
    Recursively describe the process tree using the given templates.
    :param node: tuple representing the process tree (activity or nested structure)
    :param is_root: boolean flag indicating if this is the root of the tree
    :return: string description of the process
    """
    if isinstance(node, str):
        # Base case: Atomic activity
        return node

    # Check the type of node
    node_type = node[0]
    
    if node_type == '->':
        # Sequence - describes a sequence of steps that occur one after another
        steps = [describe_process(subnode, is_root=False) for subnode in node[1:]]
        description = " ".join(f"Next, {step}." for step in steps)
        if is_root:
            # Add the starting and ending statements only for the root sequence
            description = f"{description}"
        return description
    
    elif node_type == 'X':
        # Exclusive choice with conditions and ordinal descriptions
        # Represents a decision point where only one branch is taken based on conditions
        ordinal_map = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth"]
        choices = []
        for index, condition_branch in enumerate(node[1:]):
            condition = condition_branch[0]  # The condition to check
            sub_process = condition_branch[1]  # The process to execute if condition is true
            description = describe_process(sub_process, is_root=False)
            if description == "tau":
                # Replace "tau" with "Skip and continue." if encountered
                # "tau" represents a silent/empty action in process modeling
                description = "Skip and continue."
            # Add the "In the <ordinal> procedure" format
            ordinal = ordinal_map[index] if index < len(ordinal_map) else f"{index + 1}th"
            choices.append(f"In the {ordinal} procedure, if {condition}, then {description}")
        # Return the formatted description with the number of choices
        return f"One of the {len(choices)} alternative procedures is executed: " + "; ".join(choices)
    
    
    elif node_type == '+':
        # Parallelism - represents tasks that can be executed in parallel
        parallel_steps = [describe_process(subnode, is_root=False) for subnode in node[1:]]
        return f"{len(parallel_steps)} procedures are executed in an arbitrary order: " + ", ".join(parallel_steps)
    
    elif node_type == '*':
        # Loop with conditions - represents a repetitive structure with entry and exit conditions
        loop_condition = node[1][0]  # Condition to continue the loop
        loop_process = node[1][1]    # Process to execute in the loop
        exit_condition = node[2][0]  # Condition to exit the loop
        
        # Describe the steps in the loop, excluding "tau" steps
        loop_steps = [describe_process(subnode, is_root=False) for subnode in node[1][1:] if describe_process(subnode) != "tau"]
        
        if loop_steps:
            # Generate the loop description
            loop_description = ", then ".join(loop_steps)
            first_step = loop_steps[0].split('. ')[0].lstrip("Next, ")
            # Return the formatted description for the loop
            return f"{loop_description}. If {loop_condition}, then re-perform from {first_step} to repeat. If {exit_condition}, the loop is not entered."
        else:
            return f"If {exit_condition}, the loop is not entered."
    
    elif node_type == 'tau':
        # Skip step (silent action) - represents no action
        return "tau"
    
    else:
        # Aggregation of activities (default handling for unknown cases)
        activities = [describe_process(subnode, is_root=False) for subnode in node[1:]]
        return ", then ".join(activities)



def revise_text(prompt, client):
    """
    Uses OpenAI's API to revise text based on a prompt.
    
    :param prompt: The text to be revised
    :param client: OpenAI client instance
    :return: API response containing the revised text
    """
    # Call ChatGPT API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
            {"role": "user", "content": "in one paragraph"},
            {"role": "user", "content": " Make it simpler."}
        ],
        max_tokens=300,  # Set appropriate token limit
        temperature=1,  # Control output diversity
    )
    
    # Return the API response
    return response



def replace_sentences_as_group(text, indices, new_sentence):
    """
    Replaces a group of sentences in a text with a new sentence.
    
    :param text: The original text
    :param indices: List of sentence indices to replace (1-based)
    :param new_sentence: The new sentence to insert
    :return: Modified text with specified sentences replaced
    """
    # Adjust indices from 1-based to 0-based
    indices = [i - 1 for i in indices]

    # Split text into sentences using regex
    sentences = re.split('([。！？\.!?])', text)

    # Recombine into complete sentences
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]

    # Find the range of sentences to replace
    start_idx = min(indices)
    end_idx = max(indices)

    # Check if indices are within valid range
    if start_idx < 0 or end_idx > len(sentences):
        print("Index out of sentence list range." + str(len(sentences)))
        return text
    
    # Collect sentences that aren't being replaced within the range
    temp = []
    for num in np.arange(start_idx, end_idx):
        if num not in indices:
            temp.append(sentences[num])
        
    # Replace the specified range with the preserved sentences and the new sentence
    sentences = sentences[:start_idx] + temp + [new_sentence] + sentences[end_idx+1:]

    # Rejoin sentences into a single text
    return ''.join(sentences)

class similarity_format(BaseModel):
    """
    Pydantic model for the similarity comparison results
    """
    original_description_gold_description: float
    updated_description_gold_description: float
    
def compare_similarity(original_description, updated_description, gold_description, client):
    """
    Compares the similarity between descriptions using OpenAI's API.
    
    :param original_description: The original process description
    :param updated_description: The updated process description
    :param gold_description: The reference/gold standard description
    :param client: OpenAI client instance
    :return: API response containing similarity scores
    """
    # System prompt explaining the comparison task
    prompt_0="""You are an expert in analyzing business process descriptions. Your task is to compare two descriptions (original_description and updated_description) against a reference description (gold_description) and evaluate their alignment using a scoring system. The score should range between 0 and 1, where 1 indicates perfect alignment with the Gold description, and lower scores indicate increasing deviations.

    **Evaluation Criteria:**
    1. **Core Process Alignment (Weight: 50%)**: Do the overall steps and workflow in the description match the Gold description? (1 = Perfectly matches, 0 = Significant deviations).
    2. **Sequence of Actions (Weight:50%)**: Are the sequences of specific tasks in the description consistent with the Gold description? (1 = Completely consistent, 0 = Significant deviations).


    Your task:
    1. Compare original_description vs. gold_description and updated_description vs. gold_description.
    2. Calculate the total alignment score for original_description vs. gold_description and updated_description vs. gold_description based on the weighted criteria.
    """
    # User prompt with the descriptions to compare
    prompt_1=f"""original_description: {original_description}.
             updated_description: {updated_description}.
                gold_description: {gold_description}."""
    # Output format instructions
    prompt_2="""
    **Output Format:**
    - Final Score (original_descriptionvs. gold_description): [Score]
    - Final Score (updated_description vs. gold_description): [Score]
    """

    # Call ChatGPT API
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt_0},
            {"role": "user", "content": prompt_1},
            {"role": "user", "content": prompt_2}
        ],
        max_tokens=300,  # Set appropriate token limit
        temperature=1,  # Control output diversity
        response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "relatoriginal_description_gold_description",
            "schema": similarity_format.model_json_schema()
        }
        }
    )
    
    # Return the API response
    return response

if __name__ == "__main__":
    # Devation sub process tree, replace with the output of deviation_discovery.py and Manually add decision rules 
    process_tree =('X',
    ('fries or wedges?==fries',('->', 'Ask about side dish', ('+', 'prepare fries', 'prepare wedges'))),
    ('fries or wedges?==wedges','prepare drink'))
    
    # Devation sentences, replace with the output of deviation_discovery.py
    dev_sentence_list=[5, 6, 7]
    
    # Original process description, replace with the original process description
    original_description=  """
    The robot receives an order. It then asks whether the customer wants a menu or just a burger. If he wants a burger only, that option is skipped, and the process continues. If he wants a menu, the Robot starts preparing food. During this phase, two tasks are executed at the same time: preparing the drink and asking the customer about side dish. If he want fries, it prepares the fries. If he want wedges, it prepares wedges. After that, the Robot prepares the burger, and gives enthusiastic status updates every 30 seconds. Finally delivers the order using a conveyor belt.
    """
    
    # Gold standard, replace with the description of the model
    gold_description = """
    The robot receives an order. It then asks whether the customer wants a menu or just a burger. If he wants a burger only, that option is skipped, and the process continues. If he wants a menu, the Robot starts preparing food. If wedges are chosen, the robot prepares the drink. However, if fries are preferred, it asks about side dish. Following this, both fries and wedges are prepared at same time. After that, the Robot prepares the burger, and gives enthusiastic status updates every 30 seconds. Finally delivers the order using a conveyor belt.
    """
    
    # Initialize OpenAI client
    client = OpenAI(
        # This is the default and can be omitted，replace with your own
        api_key='',
    )
    
    # Generate the process description from the tree
    description = describe_process(process_tree)

    # Create prompt for revision
    prompt = f"Minor revise the following description.: {description}."
    
    # Get the revised text from GPT-4o
    revised_text = revise_text(description, client)
    
    # Replace specified sentences in the original description with the revised text
    updated_description = replace_sentences_as_group(original_description, dev_sentence_list, revised_text.choices[0].message.content)
    print("Repaired Description:", updated_description)
    
    # Compare similarity between original, updated, and gold descriptions
    score_str = compare_similarity(original_description, updated_description, gold_description, client).choices[0].message.content

    # Parse the JSON response
    data = json.loads(score_str)

    # Extract similarity scores
    similarity_1 = data['original_description_gold_description']
    similarity_2 = data['updated_description_gold_description']
    
    # Print the similarity scores
    print(f"Similarity between original_description and gold_description: {similarity_1}")
    print(f"Similarity between updated_description and gold_description: {similarity_2}")