from collections import defaultdict
import pm4py
from pm4py.objects.process_tree.obj import ProcessTree, Operator

class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def find(self, value):
        if self.value == value:
            return self
        for child in self.children:
            result = child.find(value)
            if result:
                return result
        return None

    def print_tree(self, depth=0):
        print(" " * depth * 2 + str(self.value))
        for child in self.children:
            child.print_tree(depth + 1)
    

    def remove_tau_and_adjust(self, parent=None, index=None):
        if self.value in ('→', '∧'):
            # Remove all tau children
            self.children = [child for child in self.children if child.value != 'tau']
            # If only one child left, replace the current node with the child
            if len(self.children) == 1 and parent is not None and index is not None:
                parent.children[index] = self.children[0]
        # Recursively apply the same operation to all children
        for i, child in enumerate(self.children):
            child.remove_tau_and_adjust(self, i)

    
    def replace_redundant_nodes(self, parent=None):
        # Recursively process children first
        for child in self.children:
            child.replace_redundant_nodes(self)
        
        # Replace redundant nodes with their children
        if parent and self.value in ('→', '∧', '×') and self.value == parent.value:
            index = parent.children.index(self)
            parent.children.pop(index)
            for child in reversed(self.children):
                parent.children.insert(index, child)

    def find_parent(self, node):
        return self._find_parent_helper(node, None)

    def _find_parent_helper(self, target_node, parent):
        if self == target_node:
            return parent
        for child in self.children:
            result = child._find_parent_helper(target_node, self)
            if result:
                return result
        return None
    
    def get_layers(self):
        layers = defaultdict(list)
        self._fill_layers(layers)
        return layers

    def _fill_layers(self, layers, depth=0):
        layers[depth].append(self.value)
        for child in self.children:
            child._fill_layers(layers, depth + 1)
    def get_all_nodes(self):
        nodes = []
        self._collect_all_nodes(nodes)
        return nodes

    def _collect_all_nodes(self, nodes):
        # nodes.append(self.value)
        nodes.append(self)
        for child in self.children:
            child._collect_all_nodes(nodes)

    def to_pm4py_process_tree(self):
        def convert_node(node):
            if node.value == '→':
                pt_node = ProcessTree(operator=Operator.SEQUENCE)
            elif node.value == '×':
                pt_node = ProcessTree(operator=Operator.XOR)
            elif node.value == '∧':
                pt_node = ProcessTree(operator=Operator.PARALLEL)
            elif node.value == '↺':
                pt_node = ProcessTree(operator=Operator.LOOP)
            else:
                pt_node = ProcessTree(label=node.value)

            for child in node.children:
                pt_node.children.append(convert_node(child))

            return pt_node

        return convert_node(self)

    def find_nodes(self, values):
        # Collect all nodes with values in the provided set
        result = []
        if self.value in values:
            result.append(self)
        for child in self.children:
            result.extend(child.find_nodes(values))
        return result

    def find_lowest_common_ancestor(self, nodes):
        # Helper function to find paths to a node
        def find_path(root, target, path):
            if root is None:
                return False
            path.append(root)
            if root == target:
                return True
            for child in root.children:
                if find_path(child, target, path):
                    return True
            path.pop()
            return False

        paths = []
        for node in nodes:
            path = []
            find_path(self, node, path)
            paths.append(path)

        # Find the lowest common ancestor by comparing paths
        lca = None
        for i in range(min(len(p) for p in paths)):
            if all(p[i] == paths[0][i] for p in paths):
                lca = paths[0][i]
            else:
                break

        return lca

    # def filter_children_for_given_nodes(self, values):
    #     # Filter children to include only branches that lead to nodes with given values
    #     filtered_children = [child for child in self.children if any(node.value in values for node in child.get_all_nodes())]
    #     new_node = TreeNode(self.value)
    #     new_node.children = filtered_children
    #     return new_node
    def filter_children_to_range(self, values, tree):
        # Filter children of a sequential node to include only those from the first to the last match
        relevant_indices = [i for i, child in enumerate(self.children) if any(node.value in values for node in child.get_all_nodes())]
        if not relevant_indices:
            return None  # No children match

        start, end = min(relevant_indices), max(relevant_indices) 
        # new_node = TreeNode(self.value)
        new_node = SequentialNode()
        new_node.children = self.children[start:end+1]  # Include only children in the range [start, end]
        if start == 0 and tree.find_parent(self).value=='×':
            temp = ExclusiveChoiceNode()
            temp.children = [new_node]
            new_node = temp
        return new_node



class SequentialNode(TreeNode):
    def __init__(self):
        super().__init__('→')

class ExclusiveChoiceNode(TreeNode):
    def __init__(self):
        super().__init__('×')

class ParallelNode(TreeNode):
    def __init__(self):
        super().__init__('∧')

class RedoLoopNode(TreeNode):
    def __init__(self):
        super().__init__('↺')


def parse_process_tree(process_str):
    
    # Remove whitespace for easier processing
    process_str.strip()
    
    def parse_helper(s):
        if s.startswith(("->", "X", "+", "*")):
            if s.startswith('->'):
                node = SequentialNode()
                children_str = s[len('->') + 1:-1]  # Remove the operator and surrounding parentheses
                children = split_children(children_str)
                for child in children:
                    child_node = parse_helper(child)
                    node.add_child(child_node)
                return node
            elif s.startswith('X'):
                node = ExclusiveChoiceNode()
                children_str = s[len('X') + 1:-1]
                children = split_children(children_str)
                for child in children:
                    child_node = parse_helper(child)
                    node.add_child(child_node)
                return node
            elif s.startswith('+'):
                node = ParallelNode()
                children_str = s[len('+') + 1:-1]
                children = split_children(children_str)
                for child in children:
                    child_node = parse_helper(child)
                    node.add_child(child_node)
                return node
            elif s.startswith('*'):
                node = RedoLoopNode()
                children_str = s[len('*') + 1:-1]
                children = split_children(children_str)
                for child in children:
                    child_node = parse_helper(child)
                    node.add_child(child_node)
                return node
        else:
            return TreeNode(s.strip("'"))
    
    def split_children(s):
        # Split by commas, taking care of nested parentheses
        children = []
        balance = 0
        start = 0
        in_string = False
        for i, char in enumerate(s):
            if char == "'" and (i == 0 or s[i-1] != '\\'):
                in_string = not in_string
            if not in_string:
                if char in '({[':
                    balance += 1
                elif char in ')}]':
                    balance -= 1
                elif char == ',' and balance == 0:
                    children.append(s[start:i].strip())
                    start = i + 1
        children.append(s[start:].strip())
        return children

    return parse_helper(process_str)



# process_str = "->(->(->('check if car is registered', 'Customer comes to the Service'), *(->(->('enter car problems', +('send status updates to car owner via e-mail', 'waiting')), ->('repair done', 'pay through the app')), tau)), end)"
# tree = parse_process_tree(process_str)
# tree.print_tree()
