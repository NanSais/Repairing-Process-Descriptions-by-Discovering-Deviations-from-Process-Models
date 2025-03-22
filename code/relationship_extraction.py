from tree_classes import parse_process_tree
import pm4py
import re
from pydantic import BaseModel
from typing import List, Tuple
from openai import OpenAI
import json
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
# from utils_functions import convert_to_bpmn_new, traverse_specific, traverse_first_child_of_redo, trans_operator, intersection_node


def text_relations_extraction(text_description, key):
    class raltion_format(BaseModel):
        relationships: List[Tuple]

    class sentence_numbered_format(BaseModel):
        sentences: List[str]

    client = OpenAI(
        # This is the default and can be omitted
        api_key=key,
    )


    def numbering_sentence(prompt_0):
        # Invoke ChatGPT API
        response =  client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "user", "content": prompt_0}
            ],
            # max_tokens=300,  # Set an appropriate token limit
            temperature=1,  # Control output diversity
            response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "sentences",
                "schema": sentence_numbered_format.model_json_schema()
            }
        }
        )
        
        # Extract text returned by the API
        # revised_text = response['choices'][0]['message']['content']
        return response

    prompt_0 = f"""Start by numbering the sentences in the description in order.: {text_description}"""

    def relations_extraction(prompt_1, prompt_2, prompt_3):
        # Invoking the ChatGPT API
        response =  client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt_1},
                {"role": "user", "content": prompt_2},
                {"role": "user", "content": prompt_3}
            ],
            # max_tokens=300,  # Setting appropriate token limits
            temperature=1,  #  Control Output Diversity
            response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "relationships",
                "schema": raltion_format.model_json_schema()
            }
        }
        )
        
        # Extract text returned by the API
        # revised_text = response['choices'][0]['message']['content']
        return response

    numbered_sentences_json = numbering_sentence(prompt_0)
    numbered_sentences=json.loads(numbered_sentences_json.choices[0].message.content)['sentences']
    prompt_1 = """
    I'll show you an example of how to extract elements relationships from a process text description.

    The relation between two entities can be represented as ('HEAD element', 'TAIL element', 'S_p').

    'HEAD element', 'TAIL element' can be loop_n, exclusive_gateway_n, parallel_gateway_n or activities.  
    The difference between loop_n, exclusive_gateway_n is that loop_n will be labeled with the words reperform, go back etc. 'The process is skipped' and similar terms do not represent task(entities), but rather imply that there is no task in this branch. 'at the same time' usually implies that tasks are to be performed in parallel.
    It represents the execution order between 'HEAD element' and 'TAIL element' within the process'. Also include information on sentence position of the relations, 'S_p'.
    """
    prompt_2 = """
    For example, according to the following context: The process begins with the customer opening the E-Shop homepage and attempting to log in. The system then checks the customer's login credentials. If the login is not successful, the E-Shop Homepage is closed. If the login is successful, the customer proceeds to select a product, add it to the shopping cart, and save it. The customer checks if the product has already been selected. If the customer has not selected all desired products, they re-perform from selecting the product. Once all desired products are selected, the system prepares for payment and shipment at same time. The customer enters their payment data. Two procedures are executed in an arbitrary order. The customer then wait until the bank confirms the payment. While waiting for the payment confirmation the customer can enter its shipping address. If the shipping address is the same as the billing address, the process moves forward. If not, the customer enters a separate billing address. The system finishes processing the order. Finally, the E-Shop Homepage is closed.

    The mined relations are:
    text_relations = [
        ("opening the E-Shop homepage", "attempting to log into E-Shop", [1]),
        ("attempting to log into E-Shop", "checks the customer's login credentials", [2]),
        ("checks the customer's login credentials", "exclusive_gateway_1", [3, 4]),
        ("exclusive_gateway_1", "select a product", [4]),
        ("exclusive_gateway_1", "E-Shop Homepage is closed", [3]),
        ("select a product", "add it to the shopping cart", [4]),
        ("add it to the shopping cart", "save it", [4]),
        ("save it", "checks if the product has already been selected", [5]),
        ("checks if the product has already been selected", "loop_1", [6]),
        ("loop_1", "selecting the product", [6]),
        ("loop_1", "parallel_gateway_1", [7]),
        ("parallel_gateway_1", "prepares for payment", [7]),
        ("parallel_gateway_1", "prepares for shipment", [7]),
        ("prepares for payment", "enters their payment data", [8]),
        ("prepares for shipment", "enters their payment data", [8]),
        ("enters their payment data", "parallel_gateway_2", [9]),
        ("parallel_gateway_2", "wait until the bank confirms the payment", [10]),
        ("parallel_gateway_2", "enter its shipping address", [11]),
        ("enter its shipping address", "exclusive_gateway_2", [12, 13]),
        ("exclusive_gateway_2", "enters a separate billing address", [13]),
        ("exclusive_gateway_2", "finishes processing the order", [12]),
        ("enters a separate billing address", "finishes processing the order", [14]),
        ("wait until the bank confirms the payment", "finishes processing the order", [14]),
        ("finishes processing the order", "E-Shop Homepage is closed", [15])
    ]
    """

    prompt_3=f"""
    I would like you to extract all the entity relationships from the textual description sorted by sentence, modeled on the example just provided to you.
    {numbered_sentences}
    """


    text_relations_str = relations_extraction(prompt_1,prompt_2,prompt_3)

    # JSON formatted string
    json_str = text_relations_str.choices[0].message.content

    # Parse the JSON string
    data = json.loads(json_str)


    # Iterate over the relationships list
    return data['relationships']

def label_align(bpmn2,text_relations):
    process_str=pm4py.convert_to_process_tree(bpmn2).__repr__()
    tree = parse_process_tree(process_str)
    tree.remove_tau_and_adjust()
    tree.replace_redundant_nodes()
    entities = list({
        node.value for node in tree.get_all_nodes()
        if not re.search(r'×+|→+|↺+|∧+|tau', node.value)
    })

    # Extract unique entities from the relationship set
    relation_entities = set()
    for entity1, entity2, _ in text_relations:
        relation_entities.add(entity1)
        relation_entities.add(entity2)

    # Filter out unnecessary entities (e.g., loop_n, parallel_gateway_n, exclusive_gateway_n)
    filtered_relation_entities = {
        entity for entity in relation_entities
        if not re.search(r'loop_\d+|parallel_gateway_\d+|exclusive_gateway_\d+', entity)
    }

    # Load pretrained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

    # Encode text into vectors
    def encode_texts(texts):
        inputs = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings

    # Calculate cosine similarity matrix
    entity_embeddings = encode_texts(entities)
    relation_embeddings = encode_texts(filtered_relation_entities)
    similarity_matrix = cosine_similarity(entity_embeddings, relation_embeddings)

    # Set similarity threshold
    similarity_threshold = 0.6

    # Greedy matching algorithm to ensure strict one-to-one matching
    mapping = {}
    used_entities = set()
    used_relations = set()

    # Place all possible matching pairs and their similarity into a list
    all_pairs = []
    for i, entity in enumerate(entities):
        for j, relation in enumerate(filtered_relation_entities):
            similarity = similarity_matrix[i][j]
            if similarity >= similarity_threshold:
                all_pairs.append((similarity, entity, relation, i, j))

    # Sort all possible matching pairs in descending order of similarity
    all_pairs.sort(reverse=True, key=lambda x: x[0])

    # Handle special case matching
    loop_related_entities = set(
        entity2 for entity1, entity2, _ in text_relations if re.match(r'loop_\d+', entity1)
    )

    # Perform strict one-to-one greedy matching
    for similarity, entity, relation, entity_idx, relation_idx in all_pairs:
        # If the relation entity matches loop_n pattern, allow repeated mapping
        if relation in loop_related_entities:
            mapping[relation] = entity
            used_relations.add(relation)
        elif entity not in used_entities and relation not in used_relations:
            mapping[relation] = entity
            used_entities.add(entity)
            used_relations.add(relation)
    updated_text_relations = [
        (mapping.get(entity1, entity1), mapping.get(entity2, entity2), pos)
        for entity1, entity2, pos in text_relations
    ]
    return updated_text_relations

def eval_without_p(updated_text_relations,gold_standard):
    # Extract the subject and object of all relationships
    predicted_pairs = {(e1, e2) for e1, e2, _ in updated_text_relations}
    gold_pairs = {(e1, e2) for e1, e2, _ in gold_standard}

    # Helper function to match entities with wildcards
    def is_matching_entity(entity1, entity2):
        patterns = [r"loop_\d+", r"parallel_gateway_\d+", r"exclusive_gateway_\d+"]
        for pattern in patterns:
            if re.match(pattern, entity1) and re.match(pattern, entity2):
                return True
        return entity1 == entity2

    # Helper function to compare relations with wildcard handling
    def is_matching_relation(relation1, relation2):
        (e1_1, e2_1) = relation1
        (e1_2, e2_2) = relation2
        return is_matching_entity(e1_1, e1_2) and is_matching_entity(e2_1, e2_2)

    # Extract the subject and object of all relationships
    predicted_pairs = {(e1, e2) for e1, e2, _ in updated_text_relations}
    gold_pairs = {(e1, e2) for e1, e2, _ in gold_standard}

    # Find mismatched relations
    true_positives = set()
    false_positives = set(predicted_pairs)
    false_negatives = set(gold_pairs)

    # Compare each predicted pair and gold pair to check if it matches wildcard rules
    for pred in predicted_pairs:
        for gold in gold_pairs:
            if is_matching_relation(pred, gold):
                true_positives.add(pred)
                false_positives.discard(pred)
                false_negatives.discard(gold)
                break

    # Calculate Precision, Recall, and F1
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n Evaluation results (without p):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

def eval_with_p(updated_text_relations,gold_standard):
    # Helper function to match entities with wildcards
    def is_matching_entity(entity1, entity2):
        patterns = [r"loop_\d+", r"parallel_gateway_\d+", r"exclusive_gateway_\d+"]
        for pattern in patterns:
            if re.match(pattern, entity1) and re.match(pattern, entity2):
                return True
        return entity1 == entity2

    # Helper function to compare relations with wildcard handling and pos comparison
    def is_matching_relation(relation1, relation2):
        (e1_1, e2_1, pos1) = relation1
        (e1_2, e2_2, pos2) = relation2
        return is_matching_entity(e1_1, e1_2) and is_matching_entity(e2_1, e2_2) and pos1 == pos2

    # Extract subject, object, and pos of all relationships
    predicted_relations = {(e1, e2, tuple(pos)) for e1, e2, pos in updated_text_relations}
    gold_relations = {(e1, e2, tuple(pos)) for e1, e2, pos in gold_standard}

    # Find mismatched relations
    true_positives = set()
    false_positives = set(predicted_relations)
    false_negatives = set(gold_relations)

    # Compare each predicted pair and gold pair, checking for wildcard rules and pos matching
    for pred in predicted_relations:
        for gold in gold_relations:
            if is_matching_relation(pred, gold):
                true_positives.add(pred)
                false_positives.discard(pred)
                false_negatives.discard(gold)
                break

    # Calculate Precision, Recall, and F1
    precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
    recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n Evaluation results (with p):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")


if __name__ == "__main__":
    text_description = """
    The robot receives an order. It then asks whether the customer wants a menu or just a burger. If he wants a burger only, that option is skipped, and the process continues. If he wants a menu, the Robot starts preparing food. During this phase, two tasks are executed at the same time: preparing the drink and asking the customer about side dish. If he want fries, it prepares the fries. If he want wedges, it prepares wedges. After that, the Robot prepares the burger, and gives enthusiastic status updates every 30 seconds. Finally delivers the order using a conveyor belt.
    """
    # This is the default and can be omitted，replace with your own
    key=''
    text_relations = text_relations_extraction(text_description, key)
    bpmn2 = pm4py.read_bpmn('V_G01_model.bpmn')
    updated_text_relations=label_align(bpmn2,text_relations)
    print(updated_text_relations)
    # Gold standard relationship set
    gold_standard =[
        ('Robot receives order', 'Check whether the customer wants a menu or a burger', [1,2]),
        ('Check whether the customer wants a menu or a burger', 'exclusive_gateway_1', [3,4]),
        ('exclusive_gateway_1', 'Start preparing foods', [4]),
        ('exclusive_gateway_1', 'prepare burger', [3]),
        ('Start preparing foods', 'parallel_gateway_1', [5]),
        ('parallel_gateway_1', 'prepare drink', [5]),
        ('parallel_gateway_1', 'Ask about side dish', [5]),
        ('Ask about side dish', 'exclusive_gateway_2', [6,7]),
        ('exclusive_gateway_2', 'prepare wedges', [7]),
        ('exclusive_gateway_2', 'prepare fries', [6]),
        ('prepare wedges', 'prepare burger', [8]),
        ('prepare fries', 'prepare burger', [8]),
        ('prepare drink', 'prepare burger', [8]),
        ('prepare burger', 'give status updates', [8]),
        ('give status updates', 'deliver order using convey or belt', [9])
    ]

    eval_without_p(updated_text_relations,gold_standard)
    eval_with_p(updated_text_relations,gold_standard)
