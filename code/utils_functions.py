import pm4py
import re
def convert_to_bpmn_new(process_tree_root):
    pm4py_process_tree = process_tree_root.to_pm4py_process_tree()
    bpmn_graph = pm4py.convert_to_bpmn(pm4py_process_tree)
    return bpmn_graph

def traverse_specific(tree,node):
    nodes = []
    _traverse_specific_helper(tree, node, nodes)
    return nodes

def _traverse_specific_helper(tree, node, leaf_nodes):
    if not node.children:
        # leaf_nodes.append(node.value)
        if node.value == 'tau':
            leaf_nodes.append(tree.find_parent(node))
        else: 
            leaf_nodes.append(node)
    elif node.value == '→' and node.children:
        _traverse_specific_helper(tree, node.children[-1], leaf_nodes)
    elif node.value == '↺' and node.children:
        leaf_nodes.append(node)
    else:
        for child in node.children:
            _traverse_specific_helper(tree, child, leaf_nodes)

def traverse_first_child_of_redo(node):
    current_node = node
    while current_node:
        if not current_node.children:
            return current_node
        elif current_node.value == '→' or current_node.value == '↺':
            current_node = current_node.children[0]
        else:
            return current_node
    return None

def trans_operator(rel_string):
    pattern_1 = r'^exclusive_gateway_\d+$'
    pattern_2 = r'^parallel_gateway_\d+$'
    pattern_3 = r'^loop_\d+$'
    if re.match(pattern_1, rel_string):
        return '×'
    elif re.match(pattern_2, rel_string):
        return '∧'
    elif re.match(pattern_3, rel_string):
        return '↺'
    else:
        return None
    
def intersection_node(lists):

    intersection = set(lists[0])
    
    for lst in lists[1:]:
        intersection &= set(lst)
    
    return intersection