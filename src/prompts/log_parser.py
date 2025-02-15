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

Parse the following log message and identify the template and variables:

{log_message}"""

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
Unified Template: {{one unified template if yes. "None" if the answer is no}}""" 