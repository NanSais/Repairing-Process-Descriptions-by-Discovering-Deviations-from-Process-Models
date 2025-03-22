import pm4py
import re
from tree_classes import parse_process_tree, SequentialNode
from utils_functions import traverse_specific, traverse_first_child_of_redo, trans_operator

# Function to verify relations between two activities
def verify_activity_activity(relation, tree):
    """
    Verifies if a direct relationship exists between two activities in the process tree.
    Checks if activity_2 follows activity_1 in any sequential node of the tree.
    
    Args:
        relation: Tuple containing two activity names and their sentence numbers
        tree: The process tree to validate against
    
    Returns:
        Boolean indicating if the relation is valid in the tree
    """
    activity_1 = relation[0]
    activity_2 = relation[1]
    # Find all sequential nodes in the tree
    sequential_nodes = [node for node in tree.get_all_nodes() if isinstance(node, SequentialNode)]
    
    for seq_node in sequential_nodes:
        children_nodes = seq_node.children
        # Get all leaf nodes for each child of the sequential node
        output_nodes_list = [traverse_specific(tree, child) for child in children_nodes]

        for i, output_nodes in enumerate(output_nodes_list):
            # Check if activity_1 is in the current set of output nodes
            if activity_1 in (node.value for node in output_nodes):
                # If so, and there's a next child node, check if activity_2 is its first child
                if i < len(children_nodes) - 1:
                    next_node = children_nodes[i + 1]
                    input_node = traverse_first_child_of_redo(next_node)
                    if input_node and input_node.value == activity_2:
                        return True
    return False

# Function to verify relations between an activity and a gateway
def verify_activity_gateway(relation, tree):
    """
    Verifies if a relationship exists between an activity and a gateway in the process tree.
    
    Args:
        relation: Tuple containing activity name, gateway name, and sentence numbers
        tree: The process tree to validate against
    
    Returns:
        List containing the gateway node if relation is valid, None otherwise
    """
    activity_1 = relation[0]
    activity_2 = trans_operator(relation[1])  # Convert gateway name to its symbol representation
    sequential_nodes = [node for node in tree.get_all_nodes() if isinstance(node, SequentialNode)]

    for seq_node in sequential_nodes:
        children_nodes = seq_node.children
        output_nodes_list = [traverse_specific(tree, child) for child in children_nodes]

        for i, output_nodes in enumerate(output_nodes_list):
            for node in output_nodes:
                if node.value == activity_1:
                    if i < len(children_nodes) - 1:
                        next_node = children_nodes[i + 1]
                        input_node = traverse_first_child_of_redo(next_node)
                        if input_node and input_node.value == activity_2:
                            return [input_node]
    return None

# Function to verify relations between a gateway and an activity
def verify_gateway_activity(relation, tree):
    """
    Verifies if a relationship exists between a gateway and an activity in the process tree.
    First checks direct children, then sequential relationships.
    
    Args:
        relation: Tuple containing gateway name, activity name, and sentence numbers
        tree: The process tree to validate against
    
    Returns:
        List containing the gateway node if relation is valid, None otherwise
    """
    # First attempt: Check if activity is a direct child of the gateway
    for node in tree.get_all_nodes():
        if node.value == trans_operator(relation[0]):
            children = node.children
            for child in children:
                if traverse_first_child_of_redo(child).value == relation[1]:
                    return [node]
    
    # Second attempt: Check sequential relationships                
    activity_1 = trans_operator(relation[0])
    activity_2 = relation[1]
    sequential_nodes = [node for node in tree.get_all_nodes() if isinstance(node, SequentialNode)]

    for seq_node in sequential_nodes:
        children_nodes = seq_node.children
        output_nodes_list = [traverse_specific(tree, child) for child in children_nodes]

        for i, output_nodes in enumerate(output_nodes_list):
            for node in output_nodes:
                if node.value == activity_1:
                    if i < len(children_nodes) - 1:
                        next_node = children_nodes[i + 1]
                        input_node = traverse_first_child_of_redo(next_node)
                        if input_node and input_node.value == activity_2:
                            return [node]
    return None

# Function to verify relations between two gateways
def verify_gateway_gateway(relation, tree):
    """
    Verifies if a relationship exists between two gateways in the process tree.
    First checks direct parent-child relationships, then sequential relationships.
    
    Args:
        relation: Tuple containing two gateway names and their sentence numbers
        tree: The process tree to validate against
    
    Returns:
        List containing the two gateway nodes if relation is valid, None otherwise
    """
    # First attempt: Check direct parent-child relationship
    for node in tree.get_all_nodes():
        if node.value == trans_operator(relation[0]):
            children = node.children
            for child in children:
                if traverse_first_child_of_redo(child).value == trans_operator(relation[1]):
                    return [node, child]
    
    # Second attempt: Check sequential relationships                
    activity_1 = trans_operator(relation[0])
    activity_2 = trans_operator(relation[1])
    sequential_nodes = [node for node in tree.get_all_nodes() if isinstance(node, SequentialNode)]

    for seq_node in sequential_nodes:
        children_nodes = seq_node.children
        output_nodes_list = [traverse_specific(tree, child) for child in children_nodes]

        for i, output_nodes in enumerate(output_nodes_list):
            for node in output_nodes:
                if node.value == activity_1:
                    if i < len(children_nodes) - 1:
                        next_node = children_nodes[i + 1]
                        input_node = traverse_first_child_of_redo(next_node)
                        if input_node and input_node.value == activity_2:
                            return [node, input_node]
    return None

# Function to verify relations between a loop and an activity
def verify_loop_activity(relation, tree):
    """
    Verifies if a relationship exists between a loop operator and an activity.
    
    Args:
        relation: Tuple containing loop operator, activity name, and sentence numbers
        tree: The process tree to validate against
    
    Returns:
        List containing the loop node if relation is valid, None otherwise
    """
    # First attempt: Check direct loop-activity relationship
    for node in tree.get_all_nodes():
        if node.value == trans_operator(relation[0]):
            child = traverse_first_child_of_redo(node)
            if child.value == relation[1]:
                return [node]
    
    # Second attempt: Check sequential relationships                
    activity_1 = trans_operator(relation[0])
    activity_2 = relation[1]
    sequential_nodes = [node for node in tree.get_all_nodes() if isinstance(node, SequentialNode)]

    for seq_node in sequential_nodes:
        children_nodes = seq_node.children
        output_nodes_list = [traverse_specific(tree, child) for child in children_nodes]

        for i, output_nodes in enumerate(output_nodes_list):
            for node in output_nodes:
                if node.value == activity_1:
                    if i < len(children_nodes) - 1:
                        next_node = children_nodes[i + 1]
                        input_node = traverse_first_child_of_redo(next_node)
                        if input_node and input_node.value == activity_2:
                            return [node]
    return None

# Function to verify relations between a loop and a gateway
def verify_loop_gateway(relation, tree):
    """
    Verifies if a relationship exists between a loop operator and a gateway.
    
    Args:
        relation: Tuple containing loop operator, gateway name, and sentence numbers
        tree: The process tree to validate against
    
    Returns:
        List containing the loop and gateway nodes if relation is valid, None otherwise
    """
    # First attempt: Check direct loop-gateway relationship
    for node in tree.get_all_nodes():
        if node.value == trans_operator(relation[0]):
            child = traverse_first_child_of_redo(node)
            if child.value == trans_operator(relation[1]):
                return [node, child]
    
    # Second attempt: Check sequential relationships                
    activity_1 = trans_operator(relation[0])
    activity_2 = trans_operator(relation[1])
    sequential_nodes = [node for node in tree.get_all_nodes() if isinstance(node, SequentialNode)]

    for seq_node in sequential_nodes:
        children_nodes = seq_node.children
        output_nodes_list = [traverse_specific(tree, child) for child in children_nodes]

        for i, output_nodes in enumerate(output_nodes_list):
            for node in output_nodes:
                if node.value == activity_1:
                    if i < len(children_nodes) - 1:
                        next_node = children_nodes[i + 1]
                        input_node = traverse_first_child_of_redo(next_node)
                        if input_node and input_node.value == activity_2:
                            return [node, input_node]
    return None

# Function to verify relations between a gateway and a loop
def verify_gateway_loop(relation, tree):
    """
    Verifies if a relationship exists between a gateway and a loop operator.
    
    Args:
        relation: Tuple containing gateway name, loop operator, and sentence numbers
        tree: The process tree to validate against
    
    Returns:
        List containing the gateway and loop nodes if relation is valid, None otherwise
    """
    for node in tree.get_all_nodes():
        if node.value == trans_operator(relation[1]):
            for child in traverse_specific(tree, node.children[0]):
                if child.value == trans_operator(relation[0]):
                    return [child, node]
    return None

# Function to verify relations between an activity and a loop
def verify_activity_loop(relation, tree):
    """
    Verifies if a relationship exists between an activity and a loop operator.
    
    Args:
        relation: Tuple containing activity name, loop operator, and sentence numbers
        tree: The process tree to validate against
    
    Returns:
        List containing the loop node if relation is valid, None otherwise
    """
    for node in tree.get_all_nodes():
        if node.value == trans_operator(relation[1]):
            for child in traverse_specific(tree, node.children[0]):
                if child.value == relation[0]:
                    return [node]
    return None

# Function to verify relations between two loops
def verify_loop_loop(relation, tree):
    """
    Verifies if a relationship exists between two loop operators.
    
    Args:
        relation: Tuple containing two loop operators and their sentence numbers
        tree: The process tree to validate against
    
    Returns:
        List containing the two loop nodes if relation is valid, None otherwise
    """
    for node in tree.get_all_nodes():
        if node.value == trans_operator(relation[1]):
            for child in traverse_specific(tree, node.children[0]):
                if child.value == trans_operator(relation[0]):
                    return [child, node]
    return None

# Function to determine which verification function to use based on relation type
def determine_function(relation, tree):
    """
    Determines the appropriate verification function based on the types of entities in the relation.
    
    Args:
        relation: Tuple containing two entity names and their sentence numbers
        tree: The process tree to validate against
    
    Returns:
        Result from the appropriate verification function
    """
    entity1, entity2, _ = relation
    # Choose verification function based on entity types (activity, gateway, or loop)
    if trans_operator(entity1)==None and trans_operator(entity2)==None:
        return verify_activity_activity(relation,tree)
    elif trans_operator(entity1)==None and (trans_operator(entity2)=='×'or trans_operator(entity2)=='∧'):
        return verify_activity_gateway(relation,tree)
    elif (trans_operator(entity1)=='×'or trans_operator(entity1)=='∧') and trans_operator(entity2)==None:
        return verify_gateway_activity(relation,tree)
    elif (trans_operator(entity1)=='×'or trans_operator(entity1)=='∧') and (trans_operator(entity2)=='×'or trans_operator(entity2)=='∧'):
        return verify_gateway_gateway(relation,tree)
    elif trans_operator(entity1)=='↺' and trans_operator(entity2)==None:
        return verify_loop_activity(relation,tree)
    elif trans_operator(entity1)=='↺' and (trans_operator(entity2)=='×'or trans_operator(entity2)=='∧'):
        return verify_loop_gateway(relation,tree)
    elif (trans_operator(entity1)=='×'or trans_operator(entity1)=='∧') and trans_operator(entity2)=='↺':
        return verify_gateway_loop(relation,tree)
    elif trans_operator(entity1)==None and trans_operator(entity2)=='↺':
        return verify_activity_loop(relation,tree)
    elif trans_operator(entity1)=='↺' and trans_operator(entity2)=='↺':
        return verify_loop_loop(relation,tree)

# Functions to parse the process tree string representation
def parse_string(s):
    """
    Parses a process tree string representation into a nested tuple structure.
    
    Args:
        s: String representation of process tree
    
    Returns:
        Parsed nested tuple structure
    """
    s = s.strip()
    if s.startswith("->("):
        return parse_node(s, "->(", ")", "->")
    elif s.startswith("*("):
        return parse_node(s, "*(", ")", "*")
    elif s.startswith("+("):
        return parse_node(s, "+(", ")", "+")
    elif s.startswith("X("):
        return parse_node(s, "X(", ")", "X")
    else:
        return s.strip(" '")

def parse_node(s, start, end, symbol):
    """
    Parses a node in the process tree string.
    
    Args:
        s: String representation of the node
        start: Start delimiter
        end: End delimiter
        symbol: Operator symbol
    
    Returns:
        Parsed node as tuple
    """
    content = s[len(start):-1]
    parts = split_top_level(content)
    return (symbol, *[parse_string(part) for part in parts])

def split_top_level(s):
    """
    Splits a comma-separated string at the top level (ignoring commas in nested brackets).
    
    Args:
        s: String to split
    
    Returns:
        List of top-level parts
    """
    parts = []
    bracket_level = 0
    current_part = []
    for char in s:
        if char == '(':
            bracket_level += 1
        elif char == ')':
            bracket_level -= 1
        elif char == ',' and bracket_level == 0:
            parts.append(''.join(current_part).strip())
            current_part = []
            continue
        current_part.append(char)
    parts.append(''.join(current_part).strip())
    return parts

# Main function to discover deviations between text and model
def dev_descover(bpmn2, text_relations, gold_standard):
    """
    Discovers deviations between textual process descriptions and BPMN models.
    
    Args:
        bpmn2: Path to BPMN file
        text_relations: List of relations extracted from text
        gold_standard: List of sentence numbers with known deviations (for evaluation)
    
    Returns:
        Tuple containing evaluation metrics, list of deviation relations, and deviation subtree
    """
    # Read and parse the BPMN model
    bpmn2 = pm4py.read_bpmn(bpmn2)
    process_str = pm4py.convert_to_process_tree(bpmn2).__repr__()
    tree = parse_process_tree(process_str)
    tree.remove_tau_and_adjust()
    tree.replace_redundant_nodes()
    tree.get_all_nodes()
    
    # Extract entities from the model
    relation_entities = set()
    for node in tree.get_all_nodes():
        relation_entities.add(node.value)
    model_relation_entities = {
        entity for entity in relation_entities
        if not re.search(r'×+|→+|↺+|∧+|tau', entity)
    }

    # Extract entities from the text relations
    relation_entities = set()
    for entity1, entity2, _ in text_relations:
        relation_entities.add(entity1)
        relation_entities.add(entity2)

    text_relation_entities = {
        entity for entity in relation_entities
        if not re.search(r'loop_\d+|parallel_gateway_\d+|exclusive_gateway_\d+', entity)
    }

    # Find entities that exist in only one of the sources (potential deviations)
    text_not_in_model = text_relation_entities - model_relation_entities
    model_not_in_text = model_relation_entities - text_relation_entities
    print("Elements in text_relation_entities but not in model_relation_entities:", text_not_in_model)
    print("Elements in model_relation_entities but not in text_relation_entities:", model_not_in_text)
    
    # Remove relations with entities not in the model
    relations_to_remove = [relation for relation in text_relations if relation[0] in text_not_in_model or relation[1] in text_not_in_model]
    text_relations_0 = [relation for relation in text_relations if relation not in relations_to_remove]

    # Check activity-activity relations for deviations
    deviation_list = []
    for relation in text_relations_0:
        if trans_operator(relation[0])== None and trans_operator(relation[1])== None:
            if verify_activity_activity(relation,tree)==False:
                deviation_list.append(relation)

    # Remove activity-activity relations for next steps
    text_relations_0 = [relation for relation in text_relations_0 if not (trans_operator(relation[0])== None and trans_operator(relation[1])== None)]

    # Extract gateway entities from remaining relations
    relation_entities = set()
    for entity1, entity2, _ in text_relations_0:
        relation_entities.add(entity1)
        relation_entities.add(entity2)
    gateway_entities = {
        entity for entity in relation_entities
        if re.search(r'loop_\d+|parallel_gateway_\d+|exclusive_gateway_\d+', entity)
    }

    # Check gateway relations for deviations
    deviation_list_1 = []
    for gateway_entity in gateway_entities:
        gateway_temp = set()
        relations = [relation for relation in text_relations_0 if relation[0] == gateway_entity or relation[1] == gateway_entity]
        for relation in relations:
            if determine_function(relation,tree)==None:
                deviation_list_1.append(relation)
            else:
                index = relation.index(gateway_entity)
                return_nodes = determine_function(relation,tree)
                if len(return_nodes)==1:
                    gateway_temp.add(return_nodes[0])
                elif len(return_nodes)>1:
                    gateway_temp.add(return_nodes[index])
        # If a gateway maps to multiple nodes in the tree, mark all its relations as deviations
        if len(gateway_temp)>1:
            for relation in relations:
                deviation_list_1.append(relation)

    # Combine all deviation lists
    total_deviation_list = deviation_list + deviation_list_1 + relations_to_remove
    unique_tuples = []
    for item in total_deviation_list:
        if item not in unique_tuples:
            unique_tuples.append(item)

    # Extract sentence numbers from deviation relations
    sentence_number = {i for item in unique_tuples for i in item[2]}

    # Find all relations containing sentences with deviations
    final_deviation_relation_list = []
    for relation in text_relations:
        for i in relation[2]:
            if i in sentence_number:
                final_deviation_relation_list.append(relation)
                break

    # Extract deviation activities
    deviation_relation_entities = set()
    for entity1, entity2, _ in final_deviation_relation_list:
        deviation_relation_entities.add(entity2)
    deviation_activities = {
        entity for entity in deviation_relation_entities
        if not re.search(r'loop_\d+|parallel_gateway_\d+|exclusive_gateway_\d+', entity)
    }
    for i in model_not_in_text:
        deviation_activities.add(i)

    # Find deviation subtree (part of the process tree containing deviations)
    deviation_subtree = None
    if deviation_activities:
        # Find nodes corresponding to deviation activities
        matching_nodes = tree.find_nodes(deviation_activities)
        # Find lowest common ancestor
        lca = tree.find_lowest_common_ancestor(matching_nodes)

        if lca:
            # Adjust LCA if needed
            if lca.value not in ('×', '∧', '↺', '→'):
                lca = tree.find_parent(lca)
            if lca.value == '→':  # If the LCA is a sequential node
                # Filter sequential node to only include relevant children
                filtered_lca = lca.filter_children_to_range(deviation_activities, tree)
                deviation_subtree = filtered_lca
                nested_structure = parse_string(deviation_subtree.to_pm4py_process_tree().__repr__())
            else:
                deviation_subtree = lca
                nested_structure = parse_string(deviation_subtree.to_pm4py_process_tree().__repr__())
        else:
            print("No common ancestor found.")
            nested_structure = None
    else:
        print("No deviation activities found.")
        nested_structure = None

    # Get all activities in the deviation subtree
    temp = {
        node.value for node in deviation_subtree.get_all_nodes()
        if not re.search(r'×+|→+|↺+|∧+|tau', node.value)
    }
    for i in deviation_activities:
        temp.add(i)

    # Get sentence numbers for all relations with deviation activities
    final_deviation_sentence_number = set()
    for relation in text_relations:
        if relation[1] in temp:
            for num in relation[2]:
                final_deviation_sentence_number.add(num)

    # Evaluate against gold standard
    set_pred = set(final_deviation_sentence_number)
    set_gold = set(gold_standard)

    # Calculate metrics
    true_positives = len(set_pred & set_gold)
    false_positives = len(set_pred - set_gold)
    false_negatives = len(set_gold - set_pred)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score
    }
    return metrics, final_deviation_relation_list, nested_structure, set_pred

# Main execution block
if __name__ == "__main__":
    bpmn_path = "V_G01_gateway_order.bpmn"

    # Example text relations extracted from text (activity/gateway relations with sentence numbers)
    text_relations = [
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
    
    # Gold standard deviation sentences for evaluation
    sentence_gold_standard = [5, 6, 7]
    
    # Run deviation discovery
    metrics, final_deviation_relation_list, nested_structure, dev_sentence_list = dev_descover(bpmn_path, text_relations, sentence_gold_standard)
    
    # Print results
    print("\n Final Deviation Relation Triples:")
    for r in final_deviation_relation_list:
        print(r)

    print("Deviation sentence numbers:", dev_sentence_list)

    import pprint
    if nested_structure:
        print("\n Deviation Sub-process Tree:")
        pprint.pprint(nested_structure)
    print("\n Deviation sentences:")
    print("\n Evaluation Metrics:")
    print(f"Precision: {metrics['precision']:.2f}")
    print(f"Recall:    {metrics['recall']:.2f}")
    print(f"F1-score:  {metrics['f1']:.2f}")