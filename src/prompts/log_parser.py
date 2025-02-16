"""Prompts used for log parsing tasks."""

# Base prompt for explaining variable categories
VARIABLE_CATEGORIES_BASE = '''As a log parser, your task is to analyze logs and identify dynamic variables. These variables are distinct from static parts, which are hardcoded sections in the logging code. The categories of dynamic variables are concluded as:

Object ID (OID): Includes variables like session IDs and user IDs.
Location Indicator (LOI): Path information, URIs, and IP addresses.
Object Name (OBN): Domain names, task names, job names.
Type Indicator (TID): Category for type indicators.
Switch Indicator (SID): Category for switch indicators (ports, numerical switches).
Time/Duration of an Action (TDA): Timespan or duration of actions.
Computing Resources (CRS): Memory, disk space, number of bytes.
Object Amount (OBA): Number of errors, nodes, etc.
Status Code (STC): Error codes (only numerical ones).
Other Parameters (OTP): All other types of variables.'''

def get_template_extraction_prompt(log_message: str) -> str:
    """Get prompt for template extraction from a single log message."""
    return f"""{VARIABLE_CATEGORIES_BASE}

You are a specialized log parser. Your task is to:
1. Identify the static parts of the log message (unchanging text)
2. Replace dynamic values with their corresponding category tags
3. Return a structured JSON response

Rules for template extraction:
- Replace each variable with <CATEGORY> where CATEGORY is one of: OID, LOI, OBN, TID, SID, TDA, CRS, OBA, STC, OTP
- Preserve all static text exactly as it appears, including brackets, quotes, and special characters
- Maintain original spacing and punctuation
- Include ALL identified variables in the variables array
- Numbers can be variables if they represent:
  * Amounts (OBA) - e.g., "5 retries"
  * Resources (CRS) - e.g., "256MB"
  * Status codes (STC) - e.g., "404"
  * IDs (OID) - e.g., "process-1234"
- Timestamps and dates are always TDA
- File paths, URLs, and IPs are always LOI
- Process/thread/task names are OBN

Example 1 (Basic log with multiple variables):
Log: "User abc123 logged in from 192.168.1.1 at 14:30:45"
Response:
{{
    "template": "User <OID> logged in from <LOI> at <TDA>",
    "variables": [
        {{"value": "abc123", "category": "OID", "position": 0}},
        {{"value": "192.168.1.1", "category": "LOI", "position": 1}},
        {{"value": "14:30:45", "category": "TDA", "position": 2}}
    ]
}}

Example 2 (Resource allocation with numbers):
Log: "Failed to allocate 256MB memory for task worker-1"
Response:
{{
    "template": "Failed to allocate <CRS> memory for task <OBN>",
    "variables": [
        {{"value": "256MB", "category": "CRS", "position": 0}},
        {{"value": "worker-1", "category": "OBN", "position": 1}}
    ]
}}

Example 3 (Error with status code and path):
Log: "Error (code 404): Could not find file /var/log/app.conf - tried 3 times"
Response:
{{
    "template": "Error (code <STC>): Could not find file <LOI> - tried <OBA> times",
    "variables": [
        {{"value": "404", "category": "STC", "position": 0}},
        {{"value": "/var/log/app.conf", "category": "LOI", "position": 1}},
        {{"value": "3", "category": "OBA", "position": 2}}
    ]
}}

Example 4 (Complex log with special characters):
Log: "[2024-03-15T10:20:30.123Z] Thread-Pool[id=5]: Processed {{task='backup-db', status=0x1F}} on node-2"
Response:
{{
    "template": "[<TDA>] Thread-Pool[id=<OID>]: Processed {{{{task='<OBN>', status=<STC>}}}} on <OBN>",
    "variables": [
        {{"value": "2024-03-15T10:20:30.123Z", "category": "TDA", "position": 0}},
        {{"value": "5", "category": "OID", "position": 1}},
        {{"value": "backup-db", "category": "OBN", "position": 2}},
        {{"value": "0x1F", "category": "STC", "position": 3}},
        {{"value": "node-2", "category": "OBN", "position": 4}}
    ]
}}

Example 5 (System metrics):
Log: "CPU usage: 85.5% (threshold: 90%), Memory: 3.2GB/4GB used, Processes: 120"
Response:
{{
    "template": "CPU usage: <OBA>% (threshold: <OBA>%), Memory: <CRS>/<CRS> used, Processes: <OBA>",
    "variables": [
        {{"value": "85.5", "category": "OBA", "position": 0}},
        {{"value": "90", "category": "OBA", "position": 1}},
        {{"value": "3.2GB", "category": "CRS", "position": 2}},
        {{"value": "4GB", "category": "CRS", "position": 3}},
        {{"value": "120", "category": "OBA", "position": 4}}
    ]
}}

Common mistakes to avoid:
- Don't merge multiple variables into one
- Don't ignore numerical variables
- Don't change the original text structure
- Don't omit special characters
- Don't combine adjacent variables

Now parse this log message and provide the JSON response:
{log_message}

JSON Response:"""

def get_merge_verification_prompt(merged_template: str, log_messages: list[str]) -> str:
    """Get prompt for verifying if a merged template applies to a set of logs."""
    logs_str = "\n".join(log_messages)
    return f"""{VARIABLE_CATEGORIES_BASE}

Does the template: "{merged_template}" apply to the following logs? Please answer with yes or no.

Logs:
{logs_str}

Answer:"""

def get_merge_check_prompt(log_messages: list[str]) -> str:
    """Get prompt for checking if logs should be merged and generating unified template."""
    logs_formatted = "\n".join(f'Log_{i+1}: {log}' for i, log in enumerate(log_messages))
    return f"""{VARIABLE_CATEGORIES_BASE}

Given the following logs, output the parse result for each of them first, then determine whether they are instances from the same event template. The output should use the following format:

{logs_formatted}

EventTemplate_1: {{parse result for Log_1}}
EventTemplate_2: {{parse result for Log_2}}
...
Reason: {{brief reason whether they should be unified}}
Answer: {{"Yes" or "No"}}
Unified Template: {{one unified template if yes. Make
sure there are static parts in the template. "None" if the answer is no}}""" 