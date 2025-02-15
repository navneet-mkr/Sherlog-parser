# 🧮 LogParser-LLM Algorithm

## Overview

LogParser-LLM is an intelligent log parsing algorithm that combines traditional pattern matching with Large Language Models (LLMs) to achieve high accuracy in log template extraction and clustering.

## Algorithm Components

### 1. Data Structures

#### Prefix Tree
- Used for efficient template lookup and matching
- Each node contains:
  - Children nodes (Dict[str, PrefixTreeNode])
  - Variable flag (is_variable: bool)
  - Clusters (List[LogCluster])
  - Frequency counter

#### Log Cluster
- Represents a group of similar log messages
- Contains:
  - Template (LogTemplate)
  - Log IDs (Set[int])
  - Example messages (List[str])
  - Frequency counter

#### Log Template
- Represents the extracted pattern with variables
- Contains:
  - Template string
  - Variable types (Dict[str, VariableType])
  - Syntax variations (Set[str])
  - Frequency counter
  - Example messages

### 2. Core Algorithm Flow

```
Input: Log message
Output: (Template, Parameters)

1. Preprocessing
   - Tokenize log message into words
   
2. Prefix Tree Traversal
   - Search for existing matches
   - Track both strict and loose matches
   
3. Match Processing
   3.1 If strict match found:
       - Update cluster statistics
       - Extract parameters
       - Return existing template
       
   3.2 If no strict match:
       - Use LLM for template extraction
       - Check template pool for similar templates
       
   3.3 Template Pool Check:
       a) If similar template exists:
          - Verify merge using LLM
          - If verified, merge templates
          - Update tree with new syntax
          
       b) If no similar template:
          - Check loose matches for possible merges
          - If mergeable, create unified template
          - Otherwise, create new template
          
4. Tree Update
   - Add new path if needed
   - Update cluster information
   - Store syntax variation
```

### 3. Matching Criteria

#### Strict Matching
- Exact token count match
- Static tokens must match exactly
- Variables can match any token
- Syntax template must exist

#### Loose Matching
- Similar token count (within threshold)
- Static token similarity above threshold
- Variable position similarity
- LLM verification for merging

### 4. Template Merging

1. **Similarity Check**
   ```python
   similarity_threshold = 0.8
   len_ratio = min(len(t1), len(t2)) / max(len(t1), t2)
   if len_ratio < similarity_threshold:
       return False
   ```

2. **Token Comparison**
   - Compare static tokens
   - Align variable positions
   - Calculate similarity scores

3. **LLM Verification**
   - Generate merge verification prompt
   - Check if templates can be merged
   - Create unified template if approved

### 5. Variable Handling

#### Variable Types
- `OID`: Object IDs (session IDs, user IDs)
- `LOI`: Location Info (paths, URIs, IPs)
- `OBN`: Object Names (domains, tasks, jobs)
- `TID`: Type Indicators
- `SID`: Switch Indicators (numerical)
- `TDA`: Time/Date Actions
- `CRS`: Computing Resources
- `OBA`: Object Amounts
- `STC`: Status Codes
- `OTP`: Other Parameters

#### Position Assignment
1. Process variables with valid positions
2. Fill remaining positions
3. Create placeholders if needed
4. Sort by position

## Implementation Details

### 1. Error Handling

```python
try:
    # Process log message
    if not content:
        return TemplateExtractionError(...)
    
    # Validate structure
    if 'template' not in json_data:
        return TemplateExtractionError(...)
        
    # Process variables
    if not isinstance(variables, list):
        variables = []
except Exception as e:
    logger.error(f"Error: {str(e)}")
    return fallback_template
```

### 2. Performance Optimizations

1. **Prefix Tree**
   - O(m) lookup time (m = message length)
   - Efficient pattern matching
   - Reduced comparison overhead

2. **Template Pool**
   - Fast similarity lookups
   - Cached templates
   - Reduced LLM calls

3. **Batch Processing**
   - Group similar messages
   - Reduce API calls
   - Parallel processing support

## Usage Example

```python
parser = LogParserLLM(ollama_client, similarity_threshold=0.8)

# Parse single log
template, params = parser.parse_log(
    "User 123 logged in from 192.168.1.1",
    log_id=1
)

# Result:
# Template: "User <OID> logged in from <LOI>"
# Params: {
#   "param_1_OID": "123",
#   "param_2_LOI": "192.168.1.1"
# }
```

## Performance Metrics

The algorithm is evaluated on standard metrics:

1. **Accuracy**
   - Grouping Accuracy
   - Parsing Accuracy
   - F1 Template Accuracy

2. **Granularity**
   - Grouping Distance
   - Parsing Distance

3. **Efficiency**
   - Average Processing Time
   - Memory Usage
   - API Call Frequency

## References

1. Original LogParser paper
2. LLM integration techniques
3. Template matching algorithms
4. Variable extraction patterns 