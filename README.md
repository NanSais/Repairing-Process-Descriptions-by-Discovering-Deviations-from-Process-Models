# Repairing Process Descriptions by Discovering Deviations from Process Models

## Introduction
The implementation of the paper 'Repairing Process Descriptions by Discovering Deviations from Process Models'.

## Data Format

All experimental input files are located in the `Data` folder.
### Text Description Files (`.txt`)

Each file follows the same Python-style format with the following variables:

- `text_description` (str):  
  A paragraph describing the business process in natural language. This is the **input** for relationship extraction.

- `gold_standard` (List[Tuple[str, str, List[int]]]):  
  The expected control-flow relationships triples.

- `sentence_gold_standard` (List[int], optional):  
  A list of sentence indices manually identified as containing deviations. Used for evaluating deviation discovery.

Naming Convention:
- Files ending in `_model.txt` (e.g., `V_G01_model.txt`) are **the baseline descriptions** to be repaired.
- Files like `*_tasks_order.txt` or `*_gateway_order.txt` serve as **reference models** to help detect and fix deviations.

### BPMN Files (`.bpmn`)

Each `.txt` file is associated with one `.bpmn` file representing the formal process model
## Code Usage

All scripts are located in the `code` directory. Each script takes specific inputs and produces structured outputs as described below:

### `relationship_extraction.py`

- **Input**:
  - `text_description` (str): A paragraph describing the process in natural language.
  - `OpenAI API key`: For LLM-based extraction.

- **Output**:
  - `text_relations` (List[Tuple[str, str, List[int]]])  
    Extracted control-flow relationships in the format:  
    `("HEAD entity", "TAIL entity", [sentence numbers])`

---

### `deviation_discovery.py`

- **Input**:
  - `bpmn2` (str): Path to a BPMN file.
  - `text_relations` (List[Tuple[str, str, List[int]]]): Relations extracted from the process description (output of relationship_extraction.py).
  - `gold_standard` (List[int]): List of sentence indices with known deviations (optional, for evaluation).

- **Output**:
  - `metrics` (Dict): Precision, recall, F1-score for deviation detection.
  - `final_deviation_relation_list` (List[Tuple]): Relationships marked as deviating.
  - `nested_structure`: Nested tuple representation of the deviation sub-tree.
  - `dev_sentence_list` (Set[int]): Sentence indices containing deviations.


### `description_repair.py`

- **Input**:
  - `process_tree` (Tuple): Sub-process tree containing deviations (e.g., from `deviation_discovery.py`).
  - `dev_sentence_list` (List[int]): Sentence indices containing deviations.
  - `original_description` (str): Original full process description.
  - `gold_description` (str): Reference description for similarity comparison.

- **Output**:
  - `updated_description` (str): A revised description with deviation region replaced.
  - `similarity_1` (float): Similarity score between original and gold descriptions.
  - `similarity_2` (float): Similarity score between updated and gold descriptions.

