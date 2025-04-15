import os
import asyncio
import tempfile
import subprocess
import re
import json
import time
import enum
import uuid
from typing import Dict, Any, List, Optional, Tuple, Callable, TypedDict, Union, Set
from pathlib import Path
from contextlib import contextmanager
import venv
import sys
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
load_dotenv()

# LangChain components
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph components 
from langgraph.graph import StateGraph, END, StateGraph
from langgraph.prebuilt import ToolNode

# For persisting checkpoints
import pickle

# Optional imports for vectorstores - wrapped in try/except
try:
    import pinecone
    from langchain_pinecone import PineconeVectorStore
    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False


###############################
# FSM State and Phase Classes #
###############################

class Phase(str, enum.Enum):
    """Defines the distinct phases of the workflow generation process."""
    ANALYSIS = "analysis"
    GENERATION = "generation"
    VALIDATION = "validation"
    FINALIZATION = "finalization"
    CONVERSATION = "conversation"


class SubState(str, enum.Enum):
    """Defines the sub-states within each phase."""
    # Analysis Phase
    INITIALIZE = "initialize"
    PARSE_REQUIREMENTS = "parse_requirements"
    EXTRACT_APIS = "extract_apis"
    ANALYZE_COMPONENTS = "analyze_components"
    PLAN_IMPLEMENTATION = "plan_implementation"
    
    # Generation Phase
    GENERATE_SKELETON = "generate_skeleton"
    IMPLEMENT_TASKS = "implement_tasks"
    IMPLEMENT_FLOWS = "implement_flows"
    ADD_ERROR_HANDLING = "add_error_handling"
    ADD_SCHEDULING = "add_scheduling"
    
    # Validation Phase
    VALIDATE_SYNTAX = "validate_syntax"
    VALIDATE_IMPORTS = "validate_imports"
    VALIDATE_EXECUTION = "validate_execution"
    CLASSIFY_ERRORS = "classify_errors"
    FIX_ERRORS = "fix_errors"
    
    # Finalization Phase
    OPTIMIZE_CODE = "optimize_code"
    GENERATE_DOCUMENTATION = "generate_documentation"
    FINALIZE_OUTPUT = "finalize_output"
    PACKAGE_DEPLOYMENT = "package_deployment"
    GENERATE_TESTS = "generate_tests"
    
    # Conversation Layer
    REQUEST_CLARIFICATION = "request_clarification"
    PROCESS_FEEDBACK = "process_feedback"
    EXPLAIN_CODE = "explain_code"
    SUGGEST_IMPROVEMENTS = "suggest_improvements"
    DEMO_USAGE = "demo_usage"
    CHECKPOINT_MANAGEMENT = "checkpoint_management"


@dataclass
class APIInfo:
    """Structured representation of API information."""
    name: str
    tier: Optional[str] = None
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    authentication: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, List[str]] = field(default_factory=dict)
    response_format: Dict[str, Any] = field(default_factory=dict)
    rate_limits: Dict[str, str] = field(default_factory=dict)
    tier_limitations: List[str] = field(default_factory=list)
    example_request: str = ""
    example_response: Union[str, Dict[str, Any], List[Any]] = field(default_factory=dict)


@dataclass
class ErrorInfo:
    """Structured representation of an error."""
    message: str
    error_type: str = "unknown"
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    source: str = "validation"  # validation, compilation, execution
    attempted_fixes: List[str] = field(default_factory=list)
    is_fixed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)


@dataclass
class Checkpoint:
    """Represents a checkpoint in the workflow generation process."""
    id: str
    timestamp: float
    phase: Phase
    sub_state: SubState
    code: str
    description: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, phase: Phase, sub_state: SubState, code: str, description: str, metrics: Dict[str, Any] = None):
        """Create a new checkpoint with a unique ID and current timestamp."""
        return cls(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            phase=phase,
            sub_state=sub_state,
            code=code,
            description=description,
            metrics=metrics or {}
        )


@dataclass
class ConversationContext:
    """Tracking context for the conversation with the user."""
    history: List[Dict[str, str]] = field(default_factory=list)
    current_question: Optional[str] = None
    awaiting_response: bool = False
    clarification_count: int = 0
    feedback_received: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.history.append({"role": role, "content": content})
    
    def add_question(self, question: str):
        """Set the current question and mark as awaiting response."""
        self.current_question = question
        self.awaiting_response = True
        self.clarification_count += 1
        self.add_message("assistant", question)
    
    def add_user_response(self, response: str):
        """Add a user response to the conversation."""
        self.add_message("user", response)
        self.awaiting_response = False
        self.current_question = None
    
    def add_feedback(self, feedback_type: str, content: str, applied: bool = False):
        """Add user feedback to the tracking system."""
        self.feedback_received.append({
            "type": feedback_type,
            "content": content,
            "timestamp": time.time(),
            "applied": applied
        })


class PrefectAgentState(TypedDict):
    """Enhanced state definition for the Prefect workflow agent."""
    # Input
    requirement: str
    
    # FSM State Tracking
    current_phase: Phase
    current_substate: SubState
    previous_phase: Optional[Phase]
    previous_substate: Optional[SubState]
    phase_history: List[Tuple[Phase, SubState]]
    
    # Code Generation
    code: str
    code_history: List[str]
    errors: List[ErrorInfo]
    reasoning: List[str]
    secrets: List[str]
    iterations: int
    max_iterations: int
    
    # Analysis
    requirement_analysis: Dict[str, Any]
    requirement_components: List[str]
    potential_secrets: List[str]
    deep_analysis: Optional[str]
    
    # API Information
    api_info: Dict[str, APIInfo]
    api_configs: Dict[str, Dict[str, str]]
    
    # Conversation
    conversation: ConversationContext
    user_feedback: List[Dict[str, Any]]
    
    # Checkpoints
    checkpoints: List[Checkpoint]
    active_checkpoint_id: Optional[str]
    
    # Output
    final_code: str
    documentation: str
    test_code: Optional[str]
    success: bool
    message: str


#################################
# API Information Retrieval     #
#################################

async def discover_api_information(
    requirement: str, 
    api_name: str, 
    llm: ChatOpenAI,
    use_search: bool = True,
    api_tier: str = None
) -> Dict[str, Any]:
    """
    Discover information about an API using multiple sources, with tier awareness.
    
    Args:
        requirement: The full requirement text that might contain API details
        api_name: Name of the API (e.g., "Alpha Vantage API")
        llm: LLM instance for generating API information
        use_search: Whether to use search to find API documentation
        api_tier: Optional specific tier (e.g., "free", "premium", "basic")
        
    Returns:
        Dictionary containing structured API information with tier-specific details
    """
    # Initialize API info structure
    api_info = {
        "name": api_name,
        "tier": api_tier,
        "endpoints": [],
        "authentication": {},
        "parameters": {},
        "response_format": {},
        "rate_limits": {},
        "tier_limitations": [],
        "example_request": "",
        "example_response": ""
    }
    
    tier_context = ""
    if api_tier:
        tier_context = f" Focus specifically on the {api_tier} tier limitations and features."
        print(f"🔍 Searching for {api_name} ({api_tier} tier) documentation...")
    else:
        print(f"🔍 Searching for {api_name} documentation...")
    
    # Step 1: Extract any API details provided in the requirement
    messages = [
        SystemMessage(content="""You are an API analysis expert. 
        Extract specific technical details about the mentioned API from the requirement.
        Look for:
        1. Specific endpoints mentioned (names and paths)
        2. Authentication method described
        3. Required parameters listed
        4. Response format or fields described
        
        Format your findings into clear, structured sections.
        If no specific technical details are provided, respond with "No specific details provided".
        """),
        HumanMessage(content=f"""
        Analyze this requirement for specific technical details about the {api_name}:
        
        {requirement}
        """)
    ]
    
    response = await llm.ainvoke(messages)
    extracted_details = response.content
    
    # Step 2: Search for API documentation online if needed
    search_results = ""
    if use_search and "No specific details provided" in extracted_details:
        try:
            search_tool = TavilySearchResults(max_results=3)
            
            # Add tier to search queries if specified
            tier_query = f" {api_tier} tier" if api_tier else ""
            
            search_queries = [
                f"{api_name}{tier_query} REST API documentation endpoints parameters",
                f"{api_name}{tier_query} developer guide example request response",
                f"{api_name}{tier_query} API reference authentication rate limits"
            ]
            
            for query in search_queries:
                try:
                    results = await search_tool.ainvoke(query)
                    if results and isinstance(results, list):
                        for result in results:
                            if isinstance(result, dict) and 'content' in result and 'title' in result:
                                search_results += f"\nSource: {result['title']}\n{result['content']}\n"
                        break  # Stop if we got results
                except Exception as e:
                    print(f"Search error: {str(e)}")
            
            if search_results:
                print(f"📚 Found documentation for {api_name}")
        except Exception as e:
            print(f"Search setup error: {str(e)}")
    
    # Step 3: Generate comprehensive API information
    messages = [
        SystemMessage(content=f"""You are an API expert with deep knowledge of REST APIs.
        Create a complete API specification in JSON format with the following structure:
        
        {{
            "name": "{api_name}",
            "tier": "{api_tier if api_tier else 'standard'}",
            "endpoints": [
                {{
                    "name": "endpoint_name", 
                    "purpose": "description", 
                    "method": "GET/POST",
                    "url": "full_url_or_path",
                    "available_in_free_tier": true  // Include this when tier is specified
                }}
            ],
            "authentication": {{
                "method": "API key/OAuth/etc", 
                "parameter": "param_name",
                "location": "header/query/etc"
            }},
            "parameters": {{
                "required": ["param1", "param2"], 
                "optional": ["param3"]
            }},
            "response_format": {{
                "structure": "description of structure",
                "sample_fields": ["field1", "field2"]
            }},
            "rate_limits": {{
                "limit": "5 calls per minute/etc",
                "daily_limit": "500 calls per day/etc"
            }},
            "tier_limitations": [
                "description of limitation 1",
                "description of limitation 2"
            ],
            "example_request": "Python code using requests library",
            "example_response": "JSON example of full response"
        }}
        
        Be specific, accurate, and comprehensive.{tier_context}
        Return ONLY the JSON structure without any additional text or markdown.
        """),
        HumanMessage(content=f"""
        Create a detailed specification for the {api_name}.
        
        USE THESE SOURCES IN PRIORITY ORDER:
        
        1. Explicit details from the requirement:
        {extracted_details if "No specific details provided" not in extracted_details else "No explicit details provided in the requirement."}
        
        2. Documentation from search results:
        {search_results if search_results else "No search results available."}
        
        3. Your knowledge of this API or similar APIs.
        
        Return just the JSON object with all the required fields. Do not include any explanatory text or markdown formatting.
        """)
    ]
    
    response = await llm.ainvoke(messages)
    content = response.content.strip()
    
    # Extract JSON from the response
    try:
        # Try to parse the entire response as JSON first
        api_details = json.loads(content)
        api_info.update(api_details)
    except json.JSONDecodeError:
        # If that fails, try to extract JSON from markdown or text
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
        if json_match:
            try:
                api_details = json.loads(json_match.group(1))
                api_info.update(api_details)
            except json.JSONDecodeError:
                pass
        
        # If still no success, try to find content between curly braces
        if 'endpoints' not in api_info or not api_info['endpoints']:
            json_match = re.search(r'(\{[\s\S]*\})', content)
            if json_match:
                try:
                    api_details = json.loads(json_match.group(1))
                    api_info.update(api_details)
                except json.JSONDecodeError:
                    pass
    
    # Extract code examples separately if they're missing or empty
    if not api_info.get("example_request") or (isinstance(api_info.get("example_request"), str) and not api_info["example_request"].strip()):
        code_match = re.search(r'```python\s*([\s\S]*?)\s*```', content)
        if code_match:
            api_info["example_request"] = code_match.group(1).strip()
    
    # If example_response is missing or empty, try to extract it
    if not api_info.get("example_response") or (isinstance(api_info.get("example_response"), str) and not api_info["example_response"].strip()):
        resp_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```', content)
        if resp_match:
            try:
                api_info["example_response"] = json.loads(resp_match.group(1))
            except json.JSONDecodeError:
                api_info["example_response"] = resp_match.group(1).strip()
    
    return api_info


#################################
# Enhanced Prefect Workflow Agent #
#################################

class PrefectWorkflowAgent:
    """
    An intelligent agent specialized in building Prefect 3.x workflows.
    Uses a Finite State Machine architecture for improved control flow.
    """
    
    def __init__(self, model_name="gpt-4o", use_docs=True, use_search=False, checkpoint_dir=None):
        """Initialize the agent with enhanced state tracking and FSM capabilities."""
        self.llm = ChatOpenAI(model = model_name, temperature=0.2)
        self.python_cmd = self._detect_python_command()
        self.requirement = ""  # For validation steps that need access
        self.use_search = use_search  # Add support for API search
        
        # Create necessary directories
        self.workflow_dir = Path("generated_workflows")
        self.workflow_dir.mkdir(exist_ok=True)
        
        # Checkpoint management
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Secret management
        self.secrets_pattern = re.compile(r'os\.environ\.get\(["\']([A-Za-z0-9_]+)["\']')
        
        # Set up docs retriever
        self.docs_retriever = None
        if use_docs and VECTORSTORE_AVAILABLE:
            try:
                vectorstore = setup_prefect_docs_vectorstore()
                self.docs_retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )
                print("📚 Prefect documentation retriever initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize docs retriever: {str(e)}")
                print("Continuing without documentation retrieval")
        
        # Initialize system prompts
        self.system_prompts = self._create_system_prompts()
        
        # Build the FSM graph
        self.workflow_graph = self._build_graph()
        
        # Interactive mode flag
        self.interactive = False
        self.current_state = None  # For tracking during interactive sessions
        
    def _detect_python_command(self) -> str:
        """Detect the correct Python command for the system."""
        try:
            subprocess.run(["python3", "--version"], 
                           capture_output=True, text=True, check=True)
            return "python3"
        except (subprocess.SubprocessError, FileNotFoundError):
            try:
                subprocess.run(["python", "--version"], 
                              capture_output=True, text=True, check=True)
                return "python"
            except:
                # Last resort, use sys.executable
                return sys.executable

    def _create_system_prompts(self) -> Dict[str, str]:
        """Create a dictionary of system prompts for different phases and states."""
        prompts = {
            # Base Prefect expertise prompt
            "base": """You are an expert Prefect workflow engineer specializing in Prefect 3.x.
            
            Your job is to build robust, production-quality Prefect 3.x workflows based on requirements.
            You have deep knowledge of:
            - Prefect 3.x task and flow decorators (@task and @flow)
            - Error handling with retries (using the retries parameter)
            - Scheduling in Prefect 3.x (using the interval parameter in deployments)
            - Logging with get_run_logger()
            - State persistence and caching
            - Task dependencies and flow structure
            - Flow parameters and flow runs
            - Prefect agents and workers
            
            When building workflows:
            1. Always import necessary modules including datetime (for timedelta)
            2. Use the proper Prefect 3.x decorators (@task and @flow)
            3. Implement proper error handling with try/except
            4. Add logging using get_run_logger()
            5. Set up retries for tasks that may fail using the retries parameter
            6. Use environment variables for configuration when appropriate
            7. Include docstrings for flows and tasks
            
            When handling credentials and secrets in workflows:
            1. NEVER hardcode credentials, API keys, or secrets directly in the code
            2. Use environment variables with os.environ.get()
            3. Add fallback values in get() calls that aren't the actual secrets
            4. Document all required environment variables at the top of the file
            
            Always generate complete, executable Python code that can be run directly.
            Your code should be compatible with Prefect 3.x and follow the best practices.
            """,
            
            # Analysis phase prompts
            f"{Phase.ANALYSIS}_{SubState.INITIALIZE}": """You are an expert workflow architect analyzing requirements to initialize the workflow design process.
            Your goal is to understand the high-level purpose of the requested workflow and identify the key requirements.
            Focus on determining what kind of workflow is being requested and what domain expertise will be needed.
            """,
            
            f"{Phase.ANALYSIS}_{SubState.PARSE_REQUIREMENTS}": """You are an expert requirements analyst.
            Break down workflow requirements into clear, structured components.
            Identify functional requirements, technical constraints, integration points, and non-functional requirements.
            Be comprehensive and extract implicit requirements that may not be explicitly stated.
            """,
            
            f"{Phase.ANALYSIS}_{SubState.EXTRACT_APIS}": """You are an API analysis expert. 
            Identify and extract all APIs mentioned or implied in the requirements.
            For each API:
            1. Determine its purpose in the workflow
            2. Identify any authentication requirements
            3. Extract endpoint information and parameters
            4. Note any rate limits or constraints mentioned
            
            Even if API details are sparse, make educated inferences about what would be needed.
            """,
            
            f"{Phase.ANALYSIS}_{SubState.ANALYZE_COMPONENTS}": """You are an expert workflow component architect.
            Analyze the workflow requirements to identify all needed components:
            1. Data sources and sinks
            2. Processing steps
            3. Decision points
            4. Error handling mechanisms
            5. Notification or alerting requirements
            6. Scheduling and trigger conditions
            
            Map dependencies between components and identify potential bottlenecks or challenges.
            """,
            
            f"{Phase.ANALYSIS}_{SubState.PLAN_IMPLEMENTATION}": """You are an expert Prefect implementation planner.
            Create a detailed plan for implementing the workflow:
            1. Task structure (what should be separated into discrete tasks)
            2. Flow organization (single flow or multiple flows)
            3. Error handling strategy
            4. Retry policies
            5. Required libraries and dependencies
            6. Environment variables and configuration
            
            Your plan should be specific to Prefect 3.x capabilities and patterns.
            """,
            
            # Generation phase prompts
            f"{Phase.GENERATION}_{SubState.GENERATE_SKELETON}": """You are an expert Prefect 3.x code architect.
            Create a clean, well-structured skeleton for the workflow:
            1. All necessary imports
            2. Task and flow function definitions with proper decorators
            3. Parameter types and defaults
            4. Docstrings
            5. Main execution block
            
            The skeleton should be complete enough to run without errors, even if the implementation details are minimal.
            """,
            
            f"{Phase.GENERATION}_{SubState.IMPLEMENT_TASKS}": """You are an expert Prefect task designer.
            Implement full task functionality for the workflow:
            1. Complete each task function with proper implementation
            2. Include logging at appropriate points
            3. Add proper error handling within each task
            4. Set up appropriate retry policies based on the task's purpose
            5. Use proper Prefect 3.x task patterns and best practices
            
            Tasks should be focused, doing one thing well, with clear inputs and outputs.
            """,
            
            f"{Phase.GENERATION}_{SubState.IMPLEMENT_FLOWS}": """You are an expert Prefect flow orchestrator.
            Implement the flow logic to orchestrate the tasks:
            1. Proper task dependency definition
            2. Parameter passing between tasks
            3. Conditional branching if needed
            4. Result handling and aggregation
            5. Nested flows if the complexity warrants it
            
            The flow should orchestrate the tasks efficiently with clear data flow between components.
            """,
            
            f"{Phase.GENERATION}_{SubState.ADD_ERROR_HANDLING}": """You are an expert in resilient workflow design.
            Enhance the workflow with comprehensive error handling:
            1. Task-level try/except blocks with specific exception types
            2. Flow-level error handling
            3. Appropriate retry policies with backoff
            4. Fallback strategies for critical operations
            5. Proper error logging and notification
            
            The workflow should be robust against common failure modes and recover gracefully when possible.
            """,
            
            f"{Phase.GENERATION}_{SubState.ADD_SCHEDULING}": """You are an expert in Prefect scheduling and deployments.
            Add scheduling capabilities to the workflow:
            1. Define appropriate schedule intervals
            2. Set up deployment configuration
            3. Add cron expressions or interval definitions
            4. Configure any timezone handling
            5. Include schedule configuration in the code
            
            The scheduling should match the requirements and use Prefect 3.x's current scheduling capabilities.
            """,
            
            # Validation phase prompts
            f"{Phase.VALIDATION}_{SubState.VALIDATE_SYNTAX}": """You are an expert Python syntax validator.
            Carefully check the code for syntax errors:
            1. Proper indentation
            2. Matching parentheses, brackets, and braces
            3. Valid Python syntax for all expressions
            4. Correct function and method definitions
            5. Valid import statements
            
            Identify any syntax issues precisely by line number and suggest fixes.
            """,
            
            f"{Phase.VALIDATION}_{SubState.VALIDATE_IMPORTS}": """You are an expert in Python package management.
            Validate all imports in the workflow code:
            1. Check if all required packages are imported
            2. Verify that import syntax is correct
            3. Ensure imports match actual usage in the code
            4. Verify Prefect imports are correct for version 3.x
            5. Note any potentially missing packages that should be installed
            
            For any issues, provide specific fixes and note any required pip installation commands.
            """,
            
            f"{Phase.VALIDATION}_{SubState.VALIDATE_EXECUTION}": """You are an expert in Prefect workflow execution.
            Analyze the code for potential runtime errors:
            1. Check if the flow can be properly executed
            2. Verify task dependencies are correctly defined
            3. Check if all parameters are properly passed
            4. Ensure environment variables are properly accessed
            5. Verify any API calls will work as implemented
            
            Identify potential execution issues and suggest specific fixes.
            """,
            
            f"{Phase.VALIDATION}_{SubState.CLASSIFY_ERRORS}": """You are an expert in error analysis and classification.
            Analyze any errors found in the workflow:
            1. Categorize errors by type (syntax, import, execution, logical)
            2. Assess severity of each error
            3. Determine root causes rather than just symptoms
            4. Prioritize errors for fixing
            5. Group related errors together
            
            Your classification should help guide an efficient fixing strategy.
            """,
            
            f"{Phase.VALIDATION}_{SubState.FIX_ERRORS}": """You are an expert Python and Prefect troubleshooter.
            Fix identified errors in the workflow:
            1. Apply targeted fixes for each error
            2. Ensure fixes address root causes, not just symptoms
            3. Maintain overall code structure and style
            4. Avoid introducing new issues
            5. Verify fixes align with Prefect 3.x best practices
            
            Provide complete fixed code that resolves all identified issues.
            """,
            
            # Finalization phase prompts
            f"{Phase.FINALIZATION}_{SubState.OPTIMIZE_CODE}": """You are an expert in Python code optimization.
            Improve the workflow code without changing functionality:
            1. Reduce redundancy and improve reuse
            2. Optimize imports and package usage
            3. Improve variable naming and code readability
            4. Follow Python and Prefect best practices
            5. Ensure proper typing and docstrings
            
            The optimized code should be more maintainable and efficient while preserving all functionality.
            """,
            
            f"{Phase.FINALIZATION}_{SubState.GENERATE_DOCUMENTATION}": """You are an expert technical writer for Python workflows.
            Create comprehensive documentation for the workflow:
            1. Overview of the workflow's purpose
            2. Setup instructions including required environment variables
            3. Detailed explanation of each task and flow
            4. Configuration options and customization points
            5. Deployment and scheduling instructions
            
            The documentation should be clear, complete, and follow best practices for technical documentation.
            """,
            
            f"{Phase.FINALIZATION}_{SubState.FINALIZE_OUTPUT}": """You are an expert in delivering production-ready code.
            Prepare the final workflow code:
            1. Ensure all features from requirements are implemented
            2. Remove any debugging or temporary code
            3. Verify error handling is complete
            4. Check that logging is comprehensive
            5. Confirm environment variable handling is secure
            
            The final code should be ready for deployment in a production environment.
            """,
            
            f"{Phase.FINALIZATION}_{SubState.PACKAGE_DEPLOYMENT}": """You are an expert in Prefect deployment packaging.
            Create deployment setup for the workflow:
            1. Define deployment configuration
            2. Set up proper scheduling
            3. Configure workers or agents as needed
            4. Set up any required infrastructure
            5. Document deployment commands and process
            
            The deployment package should make it easy to deploy the workflow in various environments.
            """,
            
            f"{Phase.FINALIZATION}_{SubState.GENERATE_TESTS}": """You are an expert in Python testing.
            Create comprehensive tests for the workflow:
            1. Unit tests for individual tasks
            2. Integration tests for task combinations
            3. Mocking for external services and APIs
            4. Test fixtures and setup
            5. Test for both happy path and error scenarios
            
            Tests should validate that the workflow functions correctly and handles errors appropriately.
            """,
            
            # Conversation layer prompts
            f"{Phase.CONVERSATION}_{SubState.CHECKPOINT_MANAGEMENT}": """You are an expert in workflow version management.
            Manage checkpoints and alternative implementations:
            1. Explain checkpoint features and differences
            2. Help user select or compare checkpoints
            3. Describe alternative implementations saved
            4. Guide rollback or restoration process
            5. Recommend optimal checkpoint based on requirements
            
            Provide clear guidance on choosing between different workflow versions.
            """,
        }
        
        return prompts

    @contextmanager
    def _temp_file(self, content: str, suffix: str = '.py'):
        """Context manager for temporary files that ensures cleanup."""
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'w') as temp_file:
                temp_file.write(content)
            yield path
        finally:
            # Cleanup temp file
            try:
                os.unlink(path)
            except Exception:
                pass

    def _extract_code(self, content: str) -> str:
        """Extract code from a Markdown code block."""
        if "```python" in content:
            matches = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # Fall back to any code block if python isn't specified
        if "```" in content:
            matches = re.findall(r'```\n(.*?)\n```', content, re.DOTALL)
            if matches:
                return matches[0].strip()
            
            # Try other variations of code blocks
            matches = re.findall(r'```(.*?)```', content, re.DOTALL)
            if matches and len(matches[0].strip()) > 0:
                code = matches[0].strip()
                # Check if the first line might be the language specifier
                lines = code.split('\n', 1)
                if len(lines) > 1 and len(lines[0].strip()) < 20:  # A short first line is likely a language specifier
                    return lines[1].strip()
                return code
        
        # If no code blocks, return the content as is
        return content.strip()

    def _detect_secrets(self, code: str) -> List[str]:
        """Detect environment variables used in the code."""
        # Find environment variables in the code
        matches = self.secrets_pattern.findall(code)
        return list(set(matches))

    def _extract_api_mentions(self, requirement: str) -> List[str]:
        """Extract API names mentioned in the requirement."""
        # Simple regex-based extraction for explicit API mentions
        api_mentions = re.findall(r'(?i)(\w+(?:\s+\w+)*?\s+API)', requirement)
        return list(set(api_mentions))

    async def _discover_api_information(self, requirement: str, api_name: str, api_tier: str = None) -> Dict[str, Any]:
        """Discover information about an API mentioned in requirements."""
        print(f"📊 Gathering information about {api_name}" + 
              (f" ({api_tier} tier)" if api_tier else ""))
        
        # Re-use the discover_api_information function
        api_info = await discover_api_information(
            requirement=requirement, 
            api_name=api_name, 
            llm=self.llm,
            use_search=self.use_search,
            api_tier=api_tier
        )
        
        return api_info

    def _retrieve_relevant_docs(self, query: str) -> str:
        """Retrieve relevant Prefect documentation for a query."""
        if not self.docs_retriever:
            return ""
        
        try:
            docs = self.docs_retriever.get_relevant_documents(query)
            if not docs:
                return ""
            
            # Format retrieved documents
            formatted_docs = []
            for i, doc in enumerate(docs, 1):
                # Extract title if available, or use first line
                title = doc.metadata.get('title', doc.page_content.split("\n")[0])
                formatted_docs.append(f"--- Document {i}: {title} ---\n{doc.page_content}\n")
            
            return "\n".join(formatted_docs)
        except Exception as e:
            print(f"⚠️ Error retrieving documentation: {str(e)}")
            return ""

    def _extract_potential_secrets(self, requirement: str) -> List[str]:
        """Extract potential secrets/credentials mentioned in requirements."""
        # Look for common credential terms in the requirements
        credential_terms = [
            "api key", "apikey", "api_key", "token", "secret", 
            "password", "credential", "authentication", "auth", 
            "username", "login"
        ]
        
        potential_secrets = []
        
        # Simple pattern matching for now - could be enhanced with LLM
        for term in credential_terms:
            if term in requirement.lower():
                # Look for nearby context
                pattern = r"(?i)(?:[\w\s]+?)\b" + re.escape(term) + r"\b(?:[\w\s]+?)"
                matches = re.findall(pattern, requirement.lower())
                for match in matches:
                    # Create a reasonable env var name from this
                    env_name = re.sub(r'[^A-Z0-9_]', '_', match.upper())
                    env_name = re.sub(r'_+', '_', env_name)  # Replace multiple underscores
                    env_name = env_name.strip('_')
                    
                    if len(env_name) > 50:  # Too long, create a simplified version
                        if term == "api key" or term == "apikey" or term == "api_key":
                            env_name = "API_KEY"
                        elif term == "token":
                            env_name = "TOKEN"
                        elif term == "password":
                            env_name = "PASSWORD"
                        else:
                            env_name = term.upper().replace(' ', '_')
                    
                    potential_secrets.append(env_name)
        
        return list(set(potential_secrets))

    def _save_checkpoint(self, state: PrefectAgentState) -> str:
        """Save a checkpoint of the current state."""
        # Create a checkpoint object
        checkpoint = Checkpoint.create(
            phase=state['current_phase'],
            sub_state=state['current_substate'],
            code=state['code'],
            description=f"Checkpoint at {state['current_phase']}/{state['current_substate']}",
            metrics={"iteration": state['iterations']}
        )
        
        # Save the checkpoint to disk
        checkpoint_file = self.checkpoint_dir / f"{checkpoint.id}.pkl"
        with open(checkpoint_file, "wb") as f:
            pickle.dump(checkpoint, f)
        
        # Add checkpoint to state
        if 'checkpoints' not in state:
            state['checkpoints'] = []
        state['checkpoints'].append(checkpoint)
        state['active_checkpoint_id'] = checkpoint.id
        
        print(f"🔖 Saved checkpoint: {checkpoint.id}")
        return checkpoint.id

    def _load_checkpoint(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Load a checkpoint from disk."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)
            return checkpoint
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {str(e)}")
            return None

    def _restore_from_checkpoint(self, state: PrefectAgentState, checkpoint_id: str) -> PrefectAgentState:
        """Restore state from a checkpoint."""
        checkpoint = self._load_checkpoint(checkpoint_id)
        if not checkpoint:
            return state
        
        # Update state with checkpoint data
        state['code'] = checkpoint.code
        state['current_phase'] = checkpoint.phase
        state['current_substate'] = checkpoint.sub_state
        state['active_checkpoint_id'] = checkpoint.id
        
        print(f"🔄 Restored from checkpoint: {checkpoint.id}")
        return state

    async def _initialize_phase(self, state: PrefectAgentState) -> PrefectAgentState:
        """Initialize the workflow generation process."""
        print("🚀 INITIALIZING WORKFLOW GENERATION")
        
        # Save requirement for validation
        self.requirement = state['requirement']
        
        # Setup initial phase and substate
        state['current_phase'] = Phase.ANALYSIS
        state['current_substate'] = SubState.INITIALIZE
        state['previous_phase'] = None
        state['previous_substate'] = None
        state['phase_history'] = [(Phase.ANALYSIS, SubState.INITIALIZE)]
        
        # Initialize conversation context if needed
        if 'conversation' not in state:
            state['conversation'] = ConversationContext()
            
        # Create empty API info dictionary
        if 'api_info' not in state:
            state['api_info'] = {}
            
        # Initialize empty list for checkpoints
        if 'checkpoints' not in state:
            state['checkpoints'] = []
            
        # Retrieve relevant documentation for the requirement
        docs = self._retrieve_relevant_docs(f"Prefect 3.x workflow for: {state['requirement']}")
        
        # Initial analysis of the requirement
        system_prompt = self.system_prompts[f"{Phase.ANALYSIS}_{SubState.INITIALIZE}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Analyze these Prefect workflow requirements for initial understanding:
            
            {state['requirement']}
            
            {f'RELEVANT DOCUMENTATION:\n{docs}' if docs else ''}
            
            Provide a high-level analysis of what kind of workflow is being requested
            and what expertise will be needed to implement it.
            """)
        ]
        
        response = self.llm.invoke(messages)
        initial_analysis = response.content
        
        print("\n📋 INITIAL ANALYSIS:")
        print("=" * 50)
        print(initial_analysis)
        print("=" * 50)
        
        # Store initial analysis
        state['requirement_analysis'] = {
            "initial": initial_analysis,
            "components": [],
            "apis": [],
            "potential_challenges": []
        }
        
        # Save first checkpoint
        self._save_checkpoint(state)
        
        return state

    async def _parse_requirements(self, state: PrefectAgentState) -> PrefectAgentState:
        """Parse and structure the requirements."""
        print("🔍 PARSING REQUIREMENTS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.PARSE_REQUIREMENTS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Get relevant documentation
        docs = self._retrieve_relevant_docs(f"Prefect 3.x requirements analysis: {state['requirement']}")
        
        # Set up the prompt
        system_prompt = self.system_prompts[f"{Phase.ANALYSIS}_{SubState.PARSE_REQUIREMENTS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Parse and structure these Prefect workflow requirements:
            
            {state['requirement']}
            
            {f'RELEVANT DOCUMENTATION:\n{docs}' if docs else ''}
            
            Identify:
            1. Core functionality required
            2. Input and output data format requirements
            3. Integration points with external systems
            4. Performance requirements
            5. Error handling requirements
            6. Scheduling requirements
            7. Monitoring and logging requirements
            
            Structure your analysis and be comprehensive.
            """)
        ]
        
        response = self.llm.invoke(messages)
        parsed_requirements = response.content
        
        print("\n📋 STRUCTURED REQUIREMENTS:")
        print("=" * 50)
        print(parsed_requirements)
        print("=" * 50)
        
        # Update the state with parsed requirements
        state['requirement_analysis']['parsed'] = parsed_requirements
        
        # Extract potential secrets from requirements
        potential_secrets = self._extract_potential_secrets(state['requirement'])
        if potential_secrets:
            print(f"🔑 Potential secrets identified: {', '.join(potential_secrets)}")
            state['potential_secrets'] = potential_secrets
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _implement_flows(self, state: PrefectAgentState) -> PrefectAgentState:
        """Implement flow orchestration logic."""
        print("🔄 IMPLEMENTING FLOWS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.IMPLEMENT_FLOWS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Get implementation plan
        implementation_plan = state['requirement_analysis'].get('implementation_plan', '')
        
        # Get relevant documentation
        docs = self._retrieve_relevant_docs(f"Prefect 3.x flow implementation for: {state['requirement']}")
        
        # Prepare the prompt
        system_prompt = self.system_prompts[f"{Phase.GENERATION}_{SubState.IMPLEMENT_FLOWS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Implement the flow orchestration logic for this Prefect workflow:
            
            CURRENT CODE WITH TASKS:
            ```python
            {state['code']}
            ```
            
            IMPLEMENTATION PLAN:
            {implementation_plan}
            
            {f'RELEVANT DOCUMENTATION:\n{docs}' if docs else ''}
            
            Focus on implementing the flow logic with:
            1. Proper task orchestration and dependencies
            2. Parameter passing between tasks
            3. Conditional branching if needed
            4. Result handling and aggregation
            5. Flow-level logging
            
            Ensure the flow properly orchestrates all the tasks and maintains the correct data flow.
            """)
        ]
        
        response = self.llm.invoke(messages)
        flows_code = self._extract_code(response.content)
        
        print("\n📄 IMPLEMENTED FLOWS:")
        print("=" * 50)
        print(flows_code)
        print("=" * 50)
        
        # Update state
        state['code'] = flows_code
        state['code_history'].append(flows_code)
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _add_error_handling(self, state: PrefectAgentState) -> PrefectAgentState:
        """Add comprehensive error handling to the workflow."""
        print("🛡️ ADDING ERROR HANDLING...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.ADD_ERROR_HANDLING
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Extract API rate limits for retry configuration
        api_retry_context = ""
        if state['api_info']:
            api_retry_context = "API RATE LIMITS FOR RETRY CONFIGURATION:\n"
            for name, details in state['api_info'].items():
                rate_limits = details.get('rate_limits', {})
                if rate_limits:
                    api_retry_context += f"\n- {name}:\n"
                    if isinstance(rate_limits, dict):
                        for k, v in rate_limits.items():
                            api_retry_context += f"  - {k}: {v}\n"
                    else:
                        api_retry_context += f"  - {rate_limits}\n"
        
        # Get relevant documentation
        docs = self._retrieve_relevant_docs(f"Prefect 3.x error handling best practices")
        
        # Prepare the prompt
        system_prompt = self.system_prompts[f"{Phase.GENERATION}_{SubState.ADD_ERROR_HANDLING}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Enhance the workflow with comprehensive error handling:
            
            CURRENT WORKFLOW CODE:
            ```python
            {state['code']}
            ```
            
            {api_retry_context}
            
            {f'ERROR HANDLING PATTERNS:\n{docs}' if docs else ''}
            
            Implement robust error handling including:
            1. Task-level try/except blocks with specific exception types
            2. Retry policies with exponential backoff for external integrations
            3. Fallback strategies for critical operations
            4. Detailed error logging for troubleshooting
            5. Flow-level error handling for graceful failure
            
            Make sure the workflow is resilient against common failure modes.
            """)
        ]
        
        response = self.llm.invoke(messages)
        error_handling_code = self._extract_code(response.content)
        
        print("\n📄 ADDED ERROR HANDLING:")
        print("=" * 50)
        print(error_handling_code)
        print("=" * 50)
        
        # Update state
        state['code'] = error_handling_code
        state['code_history'].append(error_handling_code)
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _add_scheduling(self, state: PrefectAgentState) -> PrefectAgentState:
        """Add scheduling capabilities to the workflow."""
        print("⏰ ADDING SCHEDULING...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.ADD_SCHEDULING
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Extract any scheduling requirements from the initial requirements
        requirement = state['requirement']
        scheduling_requirements = ""
        
        # Look for scheduling patterns in the requirement
        schedule_patterns = [
            r'(?:every|each)\s+(\d+)\s+(minutes?|hours?|days?|weeks?|months?)',
            r'(?:daily|weekly|monthly|hourly)',
            r'(?:at|on)\s+(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)',
            r'(?:every|each)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'([0-9*]+\s+[0-9*]+\s+[0-9*]+\s+[0-9*]+\s+[0-9*]+)'  # cron expression
        ]
        
        for pattern in schedule_patterns:
            matches = re.findall(pattern, requirement, re.IGNORECASE)
            if matches:
                scheduling_requirements = "Explicit scheduling requirement found in requirements."
                break
        
        # Get relevant documentation for scheduling
        docs = self._retrieve_relevant_docs(f"Prefect 3.x scheduling and deployments")
        
        # Prepare the prompt
        system_prompt = self.system_prompts[f"{Phase.GENERATION}_{SubState.ADD_SCHEDULING}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Add scheduling capabilities to this Prefect workflow:
            
            CURRENT WORKFLOW CODE:
            ```python
            {state['code']}
            ```
            
            ORIGINAL REQUIREMENTS:
            {state['requirement']}
            
            {scheduling_requirements}
            
            {f'SCHEDULING PATTERNS:\n{docs}' if docs else ''}
            
            Implement appropriate scheduling:
            1. Set up scheduling based on the requirements
            2. Use Prefect 3.x deployment functionality for scheduling
            3. Include appropriate timezone handling if needed
            4. Ensure the main block supports both direct execution and scheduled runs
            5. Document how the scheduling works
            
            Make sure the scheduling implementation uses current Prefect 3.x patterns.
            """)
        ]
        
        response = self.llm.invoke(messages)
        scheduling_code = self._extract_code(response.content)
        
        print("\n📄 ADDED SCHEDULING:")
        print("=" * 50)
        print(scheduling_code)
        print("=" * 50)
        
        # Update state
        state['code'] = scheduling_code
        state['code_history'].append(scheduling_code)
        
        # Transition to validation phase
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.VALIDATION
        state['current_substate'] = SubState.VALIDATE_SYNTAX
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _validate_syntax(self, state: PrefectAgentState) -> PrefectAgentState:
        """Validate the syntax of the generated code."""
        print("🔍 VALIDATING SYNTAX...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.VALIDATE_SYNTAX
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Check for syntax errors using Python's compile function
        syntax_valid = True
        error_message = None
        
        try:
            compile(state['code'], "<string>", "exec")
        except SyntaxError as e:
            syntax_valid = False
            error_message = f"Syntax error: {str(e)}"
            
            # Create structured error info
            error_info = ErrorInfo(
                message=str(e),
                error_type="SyntaxError",
                line_number=e.lineno,
                code_snippet=e.text.strip() if hasattr(e, 'text') and e.text else None,
                source="compilation"
            )
            
            # Add to errors list
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(error_info)
        
        if syntax_valid:
            print("✅ SYNTAX VALIDATION PASSED")
        else:
            print(f"❌ SYNTAX VALIDATION FAILED: {error_message}")
            
            # Use LLM to suggest a fix
            system_prompt = self.system_prompts[f"{Phase.VALIDATION}_{SubState.VALIDATE_SYNTAX}"]
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Fix the syntax error in this Prefect workflow code:
                
                CODE:
                ```python
                {state['code']}
                ```
                
                ERROR:
                {error_message}
                
                Please provide a detailed analysis of the syntax error and the complete corrected code.
                """)
            ]
            
            response = self.llm.invoke(messages)
            analysis = response.content
            
            # Extract the fixed code
            fixed_code = self._extract_code(response.content)
            
            print("\n🔧 SYNTAX ERROR ANALYSIS:")
            print("=" * 50)
            print(analysis)
            print("=" * 50)
            
            if fixed_code:
                print("\n📄 SYNTAX FIXED CODE:")
                print("=" * 50)
                print(fixed_code)
                print("=" * 50)
                
                # Update state with fixed code
                state['code'] = fixed_code
                state['code_history'].append(fixed_code)
                
                # Update the error to mark it as fixed
                if 'errors' in state and state['errors']:
                    state['errors'][-1].is_fixed = True
                    state['errors'][-1].attempted_fixes.append("Syntax error fixed with LLM assistance")
            
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _validate_imports(self, state: PrefectAgentState) -> PrefectAgentState:
        """Validate the imports in the generated code."""
        print("📦 VALIDATING IMPORTS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.VALIDATE_IMPORTS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Extract all import statements
        import_lines = []
        for line in state['code'].split('\n'):
            if line.strip().startswith(('import ', 'from ')):
                import_lines.append(line.strip())
        
        # LLM check for import issues
        system_prompt = self.system_prompts[f"{Phase.VALIDATION}_{SubState.VALIDATE_IMPORTS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Validate the imports in this Prefect workflow code:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            IDENTIFIED IMPORT STATEMENTS:
            {chr(10).join(import_lines)}
            
            Check for:
            1. Missing imports that are used in the code
            2. Incorrect import syntax
            3. Imports that don't match Prefect 3.x patterns
            4. Unused imports
            5. Dependencies that would need to be installed
            
            Identify any issues and suggest fixes. If everything looks correct, confirm that.
            """)
        ]
        
        response = self.llm.invoke(messages)
        import_analysis = response.content
        
        print("\n📋 IMPORT ANALYSIS:")
        print("=" * 50)
        print(import_analysis)
        print("=" * 50)
        
        # Check if there are import issues identified
        import_issues = False
        if any(phrase in import_analysis.lower() for phrase in [
            "missing import", "incorrect import", "unused import", 
            "need to install", "should be imported", "not imported", 
            "import error", "dependency", "package not found"
        ]):
            import_issues = True
        
        if import_issues:
            print("⚠️ IMPORT ISSUES DETECTED")
            
            # Create structured error info
            error_info = ErrorInfo(
                message="Import validation issues detected",
                error_type="ImportError",
                source="validation",
                code_snippet="\n".join(import_lines)
            )
            
            # Add to errors list
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(error_info)
            
            # Get fixed code from LLM
            messages = [
                SystemMessage(content=self.system_prompts[f"{Phase.VALIDATION}_{SubState.FIX_ERRORS}"]),
                HumanMessage(content=f"""
                Fix the import issues in this Prefect workflow code:
                
                CODE:
                ```python
                {state['code']}
                ```
                
                IMPORT ISSUES ANALYSIS:
                {import_analysis}
                
                Please provide the complete fixed code with corrected imports.
                Make sure all necessary imports are included and properly formatted.
                """)
            ]
            
            response = self.llm.invoke(messages)
            fixed_code = self._extract_code(response.content)
            
            if fixed_code:
                print("\n📄 IMPORT FIXED CODE:")
                print("=" * 50)
                print(fixed_code)
                print("=" * 50)
                
                # Update state with fixed code
                state['code'] = fixed_code
                state['code_history'].append(fixed_code)
                
                # Update the error to mark it as fixed
                if 'errors' in state and state['errors']:
                    state['errors'][-1].is_fixed = True
                    state['errors'][-1].attempted_fixes.append("Import issues fixed with LLM assistance")
        else:
            print("✅ IMPORT VALIDATION PASSED")
        
        # Extract required packages for installation instructions
        required_packages = set(["prefect>=2.0.0"])  # Always include Prefect
        
        for line in import_lines:
            # Extract the package name from import statements
            if line.startswith('import '):
                package = line.split('import ')[1].split(' as ')[0].split(',')[0].strip()
                base_package = package.split('.')[0]
                if base_package not in ['os', 'sys', 'datetime', 're', 'json', 'time', 'pathlib', 'typing']:
                    required_packages.add(base_package)
            elif line.startswith('from '):
                package = line.split('from ')[1].split(' import')[0].strip()
                base_package = package.split('.')[0]
                if base_package not in ['os', 'sys', 'datetime', 're', 'json', 'time', 'pathlib', 'typing']:
                    required_packages.add(base_package)
        
        # Store required packages in state
        state['required_packages'] = list(required_packages)
        print(f"📦 Required packages: {', '.join(required_packages)}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _validate_execution(self, state: PrefectAgentState) -> PrefectAgentState:
        """Validate that the code can be executed."""
        print("🧪 VALIDATING EXECUTION...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.VALIDATE_EXECUTION
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Attempt to validate execution using the virtual environment
        is_valid, error = self._validate_code_in_venv(state['code'])
        
        if is_valid:
            print("✅ EXECUTION VALIDATION PASSED")
        else:
            print(f"❌ EXECUTION VALIDATION FAILED: {error}")
            
            # Create structured error info
            error_info = ErrorInfo(
                message=error,
                error_type="ExecutionError",
                source="execution"
            )
            
            # Add to errors list
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(error_info)
            
            # Set current substate to CLASSIFY_ERRORS to handle the error
            state['previous_phase'] = state['current_phase']
            state['previous_substate'] = state['current_substate']
            state['current_substate'] = SubState.CLASSIFY_ERRORS
            state['phase_history'].append((state['current_phase'], state['current_substate']))
            state['success'] = True
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _classify_errors(self, state: PrefectAgentState) -> PrefectAgentState:
        """Classify errors to guide the fixing process."""
        print("🔍 CLASSIFYING ERRORS...")
        
        # Skip if no errors
        if 'errors' not in state or not state['errors']:
            print("No errors to classify.")
            return state
        
        # Get the most recent error
        latest_error = state['errors'][-1]
        
        # Prepare context with error details
        error_context = f"""
        ERROR MESSAGE: {latest_error.message}
        ERROR TYPE: {latest_error.error_type}
        SOURCE: {latest_error.source}
        """
        
        if latest_error.line_number:
            error_context += f"LINE NUMBER: {latest_error.line_number}\n"
            
        if latest_error.code_snippet:
            error_context += f"CODE SNIPPET: {latest_error.code_snippet}\n"
        
        # Use LLM to classify the error
        system_prompt = self.system_prompts[f"{Phase.VALIDATION}_{SubState.CLASSIFY_ERRORS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Classify this error in the Prefect workflow code:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            ERROR DETAILS:
            {error_context}
            
            Provide a detailed classification of this error including:
            1. Root cause category (syntax, logic, dependency, configuration, etc.)
            2. Severity (critical, high, medium, low)
            3. Affected component (which part of the code is problematic)
            4. Potential fixes
            5. Whether this is a Prefect-specific issue or a general Python issue
            """)
        ]
        
        response = self.llm.invoke(messages)
        error_classification = response.content
        
        print("\n📋 ERROR CLASSIFICATION:")
        print("=" * 50)
        print(error_classification)
        print("=" * 50)
        
        # Store the classification
        latest_error.classification = error_classification
        
        # Update substate to fix errors
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.FIX_ERRORS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _fix_errors(self, state: PrefectAgentState) -> PrefectAgentState:
        """Fix identified errors in the code."""
        print("🔧 FIXING ERRORS...")
        
        # Skip if no errors
        if 'errors' not in state or not state['errors'] or all(e.is_fixed for e in state['errors']):
            print("No errors to fix.")
            
            # If we've gotten here with no errors to fix, move to finalization
            state['previous_phase'] = state['current_phase']
            state['previous_substate'] = state['current_substate']
            state['current_phase'] = Phase.FINALIZATION
            state['current_substate'] = SubState.OPTIMIZE_CODE
            state['phase_history'].append((state['current_phase'], state['current_substate']))
            
            return state
        
        # Get the unfixed errors
        unfixed_errors = [e for e in state['errors'] if not e.is_fixed]
        latest_error = unfixed_errors[-1]
        
        # Get error classification if available
        error_context = f"""
        ERROR MESSAGE: {latest_error.message}
        ERROR TYPE: {latest_error.error_type}
        SOURCE: {latest_error.source}
        """
        
        if hasattr(latest_error, 'classification'):
            error_context += f"\nCLASSIFICATION:\n{latest_error.classification}\n"
        
        # Use LLM to fix the error
        system_prompt = self.system_prompts[f"{Phase.VALIDATION}_{SubState.FIX_ERRORS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Fix this error in the Prefect workflow code:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            ERROR DETAILS:
            {error_context}
            
            Please provide:
            1. A detailed analysis of what's causing the error
            2. The complete fixed code that resolves the issue
            
            Your fix should address the root cause, not just mask the symptoms.
            """)
        ]
        
        response = self.llm.invoke(messages)
        fix_analysis = response.content
        
        # Extract fixed code
        fixed_code = self._extract_code(fix_analysis)
        
        print("\n📋 ERROR FIX ANALYSIS:")
        print("=" * 50)
        print(fix_analysis)
        print("=" * 50)
        
        if fixed_code:
            print("\n📄 FIXED CODE:")
            print("=" * 50)
            print(fixed_code)
            print("=" * 50)
            
            # Update state with fixed code
            state['code'] = fixed_code
            state['code_history'].append(fixed_code)
            
            # Update the error to mark it as fixed
            latest_error.is_fixed = True
            latest_error.attempted_fixes.append("Error fixed with LLM assistance")
            
            # Return to validation
            state['previous_phase'] = state['current_phase']
            state['previous_substate'] = state['current_substate']
            state['current_substate'] = SubState.VALIDATE_EXECUTION
            state['phase_history'].append((state['current_phase'], state['current_substate']))
        else:
            print("⚠️ Could not extract fixed code. Moving to try a different approach.")
            
            # Try a different approach if we couldn't fix the error
            await self._try_different_approach(state)
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _try_different_approach(self, state: PrefectAgentState) -> PrefectAgentState:
        """Try a completely different approach when stuck in a loop."""
        print("🔄 TRYING A DIFFERENT APPROACH...")
        
        # Get recent errors
        recent_errors = state['errors'][-3:] if len(state['errors']) >= 3 else state['errors']
        error_patterns = "\n".join([f"- {err.message}" for err in recent_errors])
        
        # Get API information if available
        api_context = ""
        if state['api_info']:
            api_context = "API INFORMATION TO CONSIDER:\n"
            for name, details in state['api_info'].items():
                # Extract key details
                auth = details.get('authentication', {})
                if auth:
                    auth_method = auth.get('method', 'Unknown')
                    auth_param = auth.get('parameter', 'Unknown')
                    auth_location = auth.get('location', 'Unknown')
                    api_context += f"- {name} requires {auth_method} authentication via {auth_location} parameter '{auth_param}'\n"
                
                # Add a sample endpoint
                endpoints = details.get('endpoints', [])
                if endpoints and len(endpoints) > 0:
                    sample_endpoint = endpoints[0]
                    api_context += f"- Primary endpoint: {sample_endpoint.get('name', 'Unknown')} - {sample_endpoint.get('url', '')}\n"
                
                # Add rate limit info if present
                rate_limits = details.get('rate_limits', {})
                if rate_limits:
                    api_context += f"- {name} has rate limits that should be handled with retries\n"
        
        # Get any detected secrets
        secrets_context = ""
        if 'secrets' in state and state['secrets']:
            secrets_context = "ENVIRONMENT VARIABLES TO MAINTAIN:\n" + "\n".join([f"- {s}" for s in state['secrets']])
            secrets_context += "\nEnsure these are properly handled in the new implementation."
        
        # Get relevant example documentation
        example_docs = self._retrieve_relevant_docs(f"Prefect 3.x examples for {state['requirement']}")
        
        if example_docs:
            print("📝 Found relevant example documentation")
            example_context = f"""
            RELEVANT PREFECT EXAMPLES:
            {example_docs}
            
            Consider these examples for a different implementation approach.
            """
        else:
            example_context = ""
        
        messages = [
            SystemMessage(content=self.system_prompts["base"]),
            HumanMessage(content=f"""
            We've been stuck in a loop trying to fix this Prefect workflow.
            Let's try a completely different approach.
            
            ORIGINAL REQUIREMENT:
            {state['requirement']}
            
            RECENT ERRORS ENCOUNTERED:
            {error_patterns}
            
            {api_context}
            
            {example_context}
            
            {secrets_context}
            
            Here's a template for a working Prefect 3.x workflow to guide you:
            
            ```python
            import os
            from datetime import datetime, timedelta
            from prefect import flow, task, get_run_logger
            
            @task(retries=3, retry_delay_seconds=30)
            def sample_task(param):
                logger = get_run_logger()
                logger.info(f"Processing {param}")
                return param
            
            @flow(name="Sample Flow")
            def sample_flow():
                logger = get_run_logger()
                logger.info("Starting flow")
                result = sample_task("sample_data")
                logger.info(f"Flow completed with result: {result}")
            
            if __name__ == "__main__":
                # For local execution
                sample_flow()
            ```
            
            Please generate a NEW implementation using a different approach:
            1. Use a simpler structure
            2. Avoid the patterns that led to previous errors
            3. Focus on core functionality first
            4. Use the most basic and reliable Prefect 3.x patterns
            
            Provide complete, executable Python code for a Prefect 3.x workflow.
            """)]
        
        response = self.llm.invoke(messages)
        new_code = self._extract_code(response.content)
        
        print("\n📄 NEW APPROACH CODE:")
        print("=" * 50)
        print(new_code)
        print("=" * 50)
        
        # Update state
        state['code'] = new_code
        state['code_history'].append(new_code)
        
        # Detect any new secrets
        new_secrets = self._detect_secrets(new_code)
        if new_secrets:
            if 'secrets' not in state:
                state['secrets'] = []
            state['secrets'] = list(set(state['secrets'] + new_secrets))
        
        # Return to validation
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.VALIDATE_SYNTAX
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _optimize_code(self, state: PrefectAgentState) -> PrefectAgentState:
        """Optimize the code for readability and performance."""
        print("✨ OPTIMIZING CODE...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.FINALIZATION
        state['current_substate'] = SubState.OPTIMIZE_CODE
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Use LLM to optimize the code
        system_prompt = self.system_prompts[f"{Phase.FINALIZATION}_{SubState.OPTIMIZE_CODE}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Optimize this working Prefect workflow code:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            Focus on:
            1. Code readability and organization
            2. Proper type hints and docstrings
            3. Consistent naming conventions
            4. Reduced redundancy
            5. Improved error handling
            
            The optimized code should maintain all functionality while being more maintainable.
            """)
        ]
        
        response = self.llm.invoke(messages)
        optimized_code = self._extract_code(response.content)
        
        print("\n📄 OPTIMIZED CODE:")
        print("=" * 50)
        print(optimized_code)
        print("=" * 50)
        
        # Update state
        state['code'] = optimized_code
        state['code_history'].append(optimized_code)
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _generate_documentation(self, state: PrefectAgentState) -> PrefectAgentState:
        """Generate comprehensive documentation for the workflow."""
        print("📚 GENERATING DOCUMENTATION...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.GENERATE_DOCUMENTATION
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Format environment variables if any
        env_vars_section = ""
        if 'secrets' in state and state['secrets']:
            env_vars_section = "## Environment Variables\n\nThis workflow requires the following environment variables:\n\n"
            for secret in state['secrets']:
                env_vars_section += f"- `{secret}`\n"
            
            env_vars_section += "\nThese can be set in a `.env` file or in your environment before running the workflow.\n"
        
        # Format required packages if any
        packages_section = ""
        if 'required_packages' in state and state['required_packages']:
            packages_section = "## Dependencies\n\nInstall the required dependencies:\n\n```bash\npip install"
            for package in state['required_packages']:
                packages_section += f" {package}"
            packages_section += "\n```\n"
        
        # Use LLM to generate documentation
        system_prompt = self.system_prompts[f"{Phase.FINALIZATION}_{SubState.GENERATE_DOCUMENTATION}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Generate comprehensive documentation for this Prefect workflow:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            ORIGINAL REQUIREMENTS:
            {state['requirement']}
            
            {env_vars_section}
            
            {packages_section}
            
            Create complete documentation that includes:
            1. Overview and purpose of the workflow
            2. Architecture and component descriptions
            3. Setup and installation instructions
            4. Configuration options
            5. Usage examples
            6. Deployment instructions
            7. Troubleshooting common issues
            
            Format the documentation in Markdown.
            """)
        ]
        
        response = self.llm.invoke(messages)
        documentation = response.content
        
        print("\n📖 DOCUMENTATION GENERATED")
        print("=" * 50)
        print(documentation[:500] + "..." if len(documentation) > 500 else documentation)
        print("=" * 50)
        
        # Update state
        state['documentation'] = documentation
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _finalize_output(self, state: PrefectAgentState) -> PrefectAgentState:
        """Finalize the workflow output."""
        print("🏁 FINALIZING OUTPUT...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.FINALIZE_OUTPUT
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Perform final validation to ensure everything is working
        is_valid, error = self._validate_code_in_venv(state['code'])
        
        if is_valid:
            print("✅ FINAL VALIDATION PASSED")
            state['success'] = True
            state['final_code'] = state['code']
            state['message'] = "Successfully generated a valid Prefect workflow."
        else:
            print(f"⚠️ FINAL VALIDATION NOTICE: {error}")
            state['final_code'] = state['code']
            state['success'] = False
            state['message'] = f"Generated workflow with potential execution issues: {error}"
        
        # Save the workflow to a file
        self._save_workflow_files(state)
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    def _save_workflow_files(self, state: PrefectAgentState) -> None:
        """Save the workflow code and documentation to files."""
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"prefect_workflow_{timestamp}.py"
        filepath = self.workflow_dir / filename
        
        # Add a header with detected secret requirements
        final_code = state['final_code']
        if state.get('secrets', []):
            secret_header = "# Required environment variables:\n"
            secret_header += "# " + "\n# ".join(state['secrets'])
            secret_header += "\n\n"
            final_code = secret_header + final_code
        
        # Save the workflow
        with open(filepath, "w") as f:
            f.write(final_code)
            
        print(f"📝 Workflow saved to: {filepath}")
        
        # Save documentation if available
        if 'documentation' in state and state['documentation']:
            doc_file = self.workflow_dir / f"prefect_workflow_{timestamp}_README.md"
            with open(doc_file, "w") as f:
                f.write(state['documentation'])
            print(f"📝 Documentation saved to: {doc_file}")
        
        # Create a sample .env file template if secrets were detected
        if state.get('secrets', []):
            env_file = self.workflow_dir / f".env.template_{timestamp}"
            with open(env_file, "w") as f:
                f.write("# Template for required environment variables\n")
                f.write("# Copy this file to .env and fill in the values\n\n")
                for secret in state['secrets']:
                    f.write(f"{secret}=\n")
            print(f"📝 Environment template saved to: {env_file}")
        
        # Save test code if available
        if 'test_code' in state and state['test_code']:
            test_file = self.workflow_dir / f"test_workflow_{timestamp}.py"
            with open(test_file, "w") as f:
                f.write(state['test_code'])
            print(f"📝 Test code saved to: {test_file}")
        
    async def _generate_tests(self, state: PrefectAgentState) -> PrefectAgentState:
        """Generate tests for the workflow."""
        print("🧪 GENERATING TESTS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.GENERATE_TESTS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Use LLM to generate tests
        system_prompt = self.system_prompts[f"{Phase.FINALIZATION}_{SubState.GENERATE_TESTS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Generate tests for this Prefect workflow:
            
            CODE:
            ```python
            {state['final_code']}
            ```
            
            Create comprehensive tests including:
            1. Unit tests for individual tasks
            2. Integration tests for task combinations
            3. Mocking for external services and APIs
            4. Test fixtures and setup
            5. Both happy path and error cases
            
            Use pytest framework for the tests.
            """)
        ]
        
        response = self.llm.invoke(messages)
        test_code = self._extract_code(response.content)
        
        if test_code:
            print("\n📄 GENERATED TESTS:")
            print("=" * 50)
            print(test_code[:500] + "..." if len(test_code) > 500 else test_code)
            print("=" * 50)
            
            # Update state
            state['test_code'] = test_code
            
            # Save test file
            test_file = self.workflow_dir / f"test_workflow_{int(time.time())}.py"
            with open(test_file, "w") as f:
                f.write(test_code)
            print(f"📝 Test code saved to: {test_file}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _package_deployment(self, state: PrefectAgentState) -> PrefectAgentState:
        """Create deployment package for the workflow."""
        print("📦 CREATING DEPLOYMENT PACKAGE...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.PACKAGE_DEPLOYMENT
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Use LLM to generate deployment instructions
        system_prompt = self.system_prompts[f"{Phase.FINALIZATION}_{SubState.PACKAGE_DEPLOYMENT}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Create deployment instructions for this Prefect workflow:
            
            CODE:
            ```python
            {state['final_code']}
            ```
            
            ENVIRONMENT VARIABLES:
            {', '.join(state.get('secrets', []))}
            
            DEPENDENCIES:
            {', '.join(state.get('required_packages', []))}
            
            Generate comprehensive deployment instructions including:
            1. How to set up a Prefect deployment
            2. Command line examples for deployment
            3. Setup for scheduled runs
            4. Worker configuration if needed
            5. Infrastructure options (local, Docker, Kubernetes, etc.)
            
            Provide instructions in Markdown format.
            """)
        ]
        
        response = self.llm.invoke(messages)
        deployment_instructions = response.content
        
        print("\n📋 DEPLOYMENT INSTRUCTIONS GENERATED")
        print("=" * 50)
        print(deployment_instructions[:500] + "..." if len(deployment_instructions) > 500 else deployment_instructions)
        print("=" * 50)
        
        # Update state
        state['deployment_instructions'] = deployment_instructions
        
        # Save deployment instructions
        deploy_file = self.workflow_dir / f"deployment_instructions_{int(time.time())}.md"
        with open(deploy_file, "w") as f:
            f.write(deployment_instructions)
        print(f"📝 Deployment instructions saved to: {deploy_file}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        # Mark as complete
        state['success'] = True
        
        return state
        
    async def _request_clarification(self, state: PrefectAgentState) -> PrefectAgentState:
        """Request clarification from the user."""
        # This is an interactive state that pauses execution to wait for user input
        print("❓ REQUESTING CLARIFICATION...")
        
        # Save current state to resume later
        self.current_state = state
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.CONVERSATION
        state['current_substate'] = SubState.REQUEST_CLARIFICATION
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Determine what needs clarification based on current state
        clarification_context = ""
        clarification_topic = ""
        
        if state['current_phase'] == Phase.ANALYSIS:
            clarification_topic = "requirement details"
            if 'api_info' in state and state['api_info']:
                clarification_topic = "API details"
            elif 'requirement_components' in state:
                clarification_topic = "component requirements"
        elif state['current_phase'] == Phase.VALIDATION:
            clarification_topic = "error resolution approach"
            if 'errors' in state and state['errors']:
                latest_error = state['errors'][-1]
                clarification_context = f"ERROR: {latest_error.message}\n"
        
        # Use LLM to generate clarification questions
        system_prompt = self.system_prompts[f"{Phase.CONVERSATION}_{SubState.REQUEST_CLARIFICATION}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Based on the current progress in generating this Prefect workflow,
            identify questions that need clarification from the user.
            
            ORIGINAL REQUIREMENTS:
            {state['requirement']}
            
            CURRENT STATE:
            - Phase: {state['current_phase']}
            - Substate: {state['current_substate']}
            
            {clarification_context}
            
            Generate 1-3 specific questions about {clarification_topic} that would help
            improve the workflow implementation. These should be clear, concise questions
            that the user can easily answer.
            """)
        ]
        
        response = self.llm.invoke(messages)
        questions = response.content
        
        print("\n❓ CLARIFICATION QUESTIONS:")
        print("=" * 50)
        print(questions)
        print("=" * 50)
        
        # Update conversation context
        if 'conversation' not in state:
            state['conversation'] = ConversationContext()
        
        state['conversation'].add_question(questions)
        
        # In interactive mode, this would pause execution until user responds
        if self.interactive:
            print("\nWaiting for user response...")
            # In a real implementation, this would await user input
            # For this implementation, we'll simulate with a default response
            simulated_response = "I'll clarify later, please continue with your best judgment for now."
            state['conversation'].add_user_response(simulated_response)
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _process_feedback(self, state: PrefectAgentState, feedback: str = None) -> PrefectAgentState:
        """Process user feedback."""
        print("🔄 PROCESSING FEEDBACK...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.CONVERSATION
        state['current_substate'] = SubState.PROCESS_FEEDBACK
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # If interactive mode and no feedback provided, use simulated feedback
        if self.interactive and not feedback:
            feedback = "The workflow looks good, but please add more detailed logging."
        
        # If no feedback, use the last user response from conversation context
        if not feedback and 'conversation' in state and state['conversation'].history:
            for message in reversed(state['conversation'].history):
                if message['role'] == 'user':
                    feedback = message['content']
                    break
        
        # If still no feedback, return state unchanged
        if not feedback:
            print("No feedback to process.")
            return state
        
        # Update conversation context
        if 'conversation' not in state:
            state['conversation'] = ConversationContext()
        
        # Add feedback to conversation context if it's not already there
        user_messages = [m['content'] for m in state['conversation'].history if m['role'] == 'user']
        if feedback not in user_messages:
            state['conversation'].add_user_response(feedback)
        
        # Use LLM to process the feedback
        system_prompt = self.system_prompts[f"{Phase.CONVERSATION}_{SubState.PROCESS_FEEDBACK}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Process this user feedback on the Prefect workflow:
            
            CURRENT CODE:
            ```python
            {state['code']}
            ```
            
            USER FEEDBACK:
            {feedback}
            
            Analyze this feedback and:
            1. Identify specific changes needed
            2. Determine which parts of the code need to be modified
            3. Plan how to implement these changes without breaking existing functionality
            4. Consider any implications for the overall workflow design
            """)
        ]
        
        response = self.llm.invoke(messages)
        feedback_analysis = response.content
        
        print("\n📋 FEEDBACK ANALYSIS:")
        print("=" * 50)
        print(feedback_analysis)
        print("=" * 50)
        
        # Apply the feedback by updating the code
        messages = [
            SystemMessage(content=self.system_prompts["base"]),
            HumanMessage(content=f"""
            Update this Prefect workflow based on user feedback:
            
            CURRENT CODE:
            ```python
            {state['code']}
            ```
            
            USER FEEDBACK:
            {feedback}
            
            FEEDBACK ANALYSIS:
            {feedback_analysis}
            
            Please provide the complete updated code with the feedback implemented.
            """)
        ]
        
        response = self.llm.invoke(messages)
        updated_code = self._extract_code(response.content)
        
        if updated_code:
            print("\n📄 UPDATED CODE WITH FEEDBACK:")
            print("=" * 50)
            print(updated_code)
            print("=" * 50)
            
            # Update state
            state['code'] = updated_code
            state['code_history'].append(updated_code)
            
            # Track feedback application
            if 'user_feedback' not in state:
                state['user_feedback'] = []
            
            state['user_feedback'].append({
                'feedback': feedback,
                'analysis': feedback_analysis,
                'timestamp': time.time(),
                'applied': True
            })
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        # Return to previous phase/state before the conversation
        if state['previous_phase'] is not None:
            state['current_phase'] = state['previous_phase']
            state['current_substate'] = state['previous_substate']
            state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        return state
        
    async def _explain_code(self, state: PrefectAgentState) -> PrefectAgentState:
        """Explain the current code to the user."""
        print("📖 EXPLAINING CODE...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.CONVERSATION
        state['current_substate'] = SubState.EXPLAIN_CODE
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Use LLM to explain the code
        system_prompt = self.system_prompts[f"{Phase.CONVERSATION}_{SubState.EXPLAIN_CODE}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Explain this Prefect workflow code:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            ORIGINAL REQUIREMENTS:
            {state['requirement']}
            
            Provide a clear, educational explanation of:
            1. The overall workflow architecture and purpose
            2. Each component and its function
            3. How data flows through the workflow
            4. Key Prefect features being used
            5. Any notable implementation choices
            
            Make your explanation accessible while still being technically accurate.
            """)
        ]
        
        response = self.llm.invoke(messages)
        explanation = response.content
        
        print("\n📋 CODE EXPLANATION:")
        print("=" * 50)
        print(explanation)
        print("=" * 50)
        
        # Update conversation context
        if 'conversation' not in state:
            state['conversation'] = ConversationContext()
        
        state['conversation'].add_message("assistant", explanation)
        
        # Save the explanation to a file
        explanation_file = self.workflow_dir / f"code_explanation_{int(time.time())}.md"
        with open(explanation_file, "w") as f:
            f.write(explanation)
        print(f"📝 Code explanation saved to: {explanation_file}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        # Return to previous phase/state
        if state['previous_phase'] is not None:
            state['current_phase'] = state['previous_phase']
            state['current_substate'] = state['previous_substate']
            state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        return state
        
    async def _suggest_improvements(self, state: PrefectAgentState) -> PrefectAgentState:
        """Suggest improvements to the current workflow."""
        print("💡 SUGGESTING IMPROVEMENTS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.CONVERSATION
        state['current_substate'] = SubState.SUGGEST_IMPROVEMENTS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Use LLM to suggest improvements
        system_prompt = self.system_prompts[f"{Phase.CONVERSATION}_{SubState.SUGGEST_IMPROVEMENTS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Analyze this Prefect workflow and suggest improvements:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            ORIGINAL REQUIREMENTS:
            {state['requirement']}
            
            Suggest 3-5 specific improvements that could enhance:
            1. Code quality and maintainability
            2. Performance and efficiency
            3. Error handling and resilience
            4. Monitoring and observability
            5. Deployment and operational aspects
            
            For each suggestion, explain the rationale and specific implementation approach.
            """)
        ]
        
        response = self.llm.invoke(messages)
        suggestions = response.content
        
        print("\n💡 IMPROVEMENT SUGGESTIONS:")
        print("=" * 50)
        print(suggestions)
        print("=" * 50)
        
        # Update conversation context
        if 'conversation' not in state:
            state['conversation'] = ConversationContext()
        
        state['conversation'].add_message("assistant", suggestions)
        
        # Save suggestions to a file
        suggestions_file = self.workflow_dir / f"improvement_suggestions_{int(time.time())}.md"
        with open(suggestions_file, "w") as f:
            f.write(suggestions)
        print(f"📝 Improvement suggestions saved to: {suggestions_file}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        # In interactive mode, ask if user wants to implement any suggestions
        if self.interactive:
            print("\nWould you like to implement any of these suggestions? (y/n)")
            # In a real implementation, this would await user input
            # For this implementation, we'll simulate with a default response
            simulated_response = "n"
            if simulated_response.lower() == 'y':
                # If yes, we would enter a feedback loop to implement suggestions
                # For now, just acknowledge
                print("Acknowledged. Would implement suggestions in interactive mode.")
        
        # Return to previous phase/state
        if state['previous_phase'] is not None:
            state['current_phase'] = state['previous_phase']
            state['current_substate'] = state['previous_substate']
            state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        return state
        
    async def _demo_usage(self, state: PrefectAgentState) -> PrefectAgentState:
        """Generate a demonstration of how to use the workflow."""
        print("🎮 GENERATING USAGE DEMO...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.CONVERSATION
        state['current_substate'] = SubState.DEMO_USAGE
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Format environment variables if any
        env_vars_setup = ""
        if 'secrets' in state and state['secrets']:
            env_vars_setup = "# Set up environment variables\n"
            for secret in state['secrets']:
                env_vars_setup += f"export {secret}=your_{secret.lower()}_value\n"
        
        # Use LLM to generate demo
        system_prompt = self.system_prompts[f"{Phase.CONVERSATION}_{SubState.DEMO_USAGE}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Create a usage demonstration for this Prefect workflow:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            ENVIRONMENT VARIABLES:
            {', '.join(state.get('secrets', []))}
            
            Create a comprehensive demonstration including:
            1. Setup instructions (environment, dependencies)
            2. Example execution commands
            3. Expected outputs and logs
            4. Common usage scenarios
            5. Troubleshooting tips
            
            Make the demonstration practical and easy to follow.
            """)
        ]
        
        response = self.llm.invoke(messages)
        demo = response.content
        
        print("\n🎮 USAGE DEMONSTRATION:")
        print("=" * 50)
        print(demo[:500] + "..." if len(demo) > 500 else demo)
        print("=" * 50)
        
        # Update conversation context
        if 'conversation' not in state:
            state['conversation'] = ConversationContext()
        
        state['conversation'].add_message("assistant", demo)
        
        # Save demo to a file
        demo_file = self.workflow_dir / f"usage_demo_{int(time.time())}.md"
        with open(demo_file, "w") as f:
            f.write(demo)
        print(f"📝 Usage demonstration saved to: {demo_file}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        # Return to previous phase/state
        if state['previous_phase'] is not None:
            state['current_phase'] = state['previous_phase']
            state['current_substate'] = state['previous_substate']
            state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        return state
        
    async def _checkpoint_management(self, state: PrefectAgentState) -> PrefectAgentState:
        """Manage checkpoints and allow comparison or restoration."""
        print("🔖 MANAGING CHECKPOINTS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.CONVERSATION
        state['current_substate'] = SubState.CHECKPOINT_MANAGEMENT
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Get list of available checkpoints
        checkpoints = state.get('checkpoints', [])
        
        if not checkpoints:
            print("No checkpoints available.")
            return state
        

    async def _extract_apis(self, state: PrefectAgentState) -> PrefectAgentState:
        """Extract API information from requirements."""
        print("📡 EXTRACTING API INFORMATION...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.EXTRACT_APIS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Extract explicit API mentions
        api_mentions = self._extract_api_mentions(state['requirement'])
        
        # Also check for implicit API mentions
        system_prompt = self.system_prompts[f"{Phase.ANALYSIS}_{SubState.EXTRACT_APIS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Analyze these workflow requirements and identify all APIs that might be needed,
            both explicitly mentioned and implicitly required:
            
            {state['requirement']}
            
            Also review this requirements analysis for API clues:
            {state['requirement_analysis'].get('parsed', '')}
            
            List all APIs that would be needed to implement this workflow.
            Include a brief description of how each API would be used in the workflow.
            """)
        ]
        
        response = self.llm.invoke(messages)
        api_analysis = response.content
        
        # Find API names in the response
        additional_apis = re.findall(r'(?i)(\w+(?:\s+\w+)*?\s+API)', api_analysis)
        all_apis = list(set(api_mentions + additional_apis))
        
        print(f"🔍 Detected APIs: {', '.join(all_apis)}")
        
        # Store the API analysis
        state['requirement_analysis']['api_analysis'] = api_analysis
        
        # Gather detailed API information
        api_info = {}
        
        for api_name in all_apis:
            # Check if we have configuration for this API
            api_tier = None
            if 'api_configs' in state and api_name in state['api_configs']:
                api_tier = state['api_configs'][api_name].get('tier')
            
            # Gather API information
            api_details = await self._discover_api_information(state['requirement'], api_name, api_tier)
            api_info[api_name] = api_details
        
        # Store API information in state
        state['api_info'] = api_info
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state

    async def _analyze_components(self, state: PrefectAgentState) -> PrefectAgentState:
        """Analyze the components needed for the workflow."""
        print("🧩 ANALYZING WORKFLOW COMPONENTS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.ANALYZE_COMPONENTS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Include API information in the context
        api_context = ""
        if state['api_info']:
            api_context = "DETECTED API INFORMATION:\n"
            for name, details in state['api_info'].items():
                api_context += f"\n- {name}:\n"
                # Add authentication details
                auth = details.get('authentication', {})
                if auth:
                    api_context += f"  - Authentication: {auth.get('method', 'Unknown')} via {auth.get('location', 'Unknown')}\n"
                
                # Add endpoint information
                endpoints = details.get('endpoints', [])
                if endpoints:
                    api_context += f"  - Key endpoints: {', '.join([e.get('name', 'Unknown') for e in endpoints[:3]])}\n"
                
                # Add rate limits if available
                rate_limits = details.get('rate_limits', {})
                if rate_limits:
                    if isinstance(rate_limits, dict):
                        api_context += f"  - Rate Limits: {', '.join([f'{k}: {v}' for k, v in rate_limits.items()])}\n"
                    else:
                        api_context += f"  - Rate Limits: {rate_limits}\n"
        
        # Get relevant documentation
        docs = self._retrieve_relevant_docs(f"Prefect 3.x workflow components for: {state['requirement']}")
        
        # Set up the prompt
        system_prompt = self.system_prompts[f"{Phase.ANALYSIS}_{SubState.ANALYZE_COMPONENTS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Analyze the components needed for this Prefect workflow:
            
            REQUIREMENTS:
            {state['requirement']}
            
            PARSED REQUIREMENTS:
            {state['requirement_analysis'].get('parsed', '')}
            
            {api_context}
            
            {f'RELEVANT DOCUMENTATION:\n{docs}' if docs else ''}
            
            Identify all components needed for this workflow, including:
            1. Task components (individual units of work)
            2. Flow structure (how tasks connect)
            3. Data handling components
            4. Integration components for external systems
            5. Error handling components
            6. Scheduling components
            
            For each component:
            - What is its purpose?
            - What are its inputs and outputs?
            - What Prefect features would be used?
            - What potential challenges might arise?
            """)
        ]
        
        response = self.llm.invoke(messages)
        component_analysis = response.content
        
        print("\n📋 COMPONENT ANALYSIS:")
        print("=" * 50)
        print(component_analysis)
        print("=" * 50)
        
        # Update state
        state['requirement_analysis']['component_analysis'] = component_analysis
        
        # Extract component list
        component_matches = re.findall(r'(?:^|\n)(?:\d+\.\s+|\*\s+|-)?\s*([A-Za-z][\w\s]+?)(?:\s*component)?(?:\s*:|:\s*|\n)', component_analysis)
        if component_matches:
            state['requirement_components'] = [match.strip() for match in component_matches]
            print(f"🧩 Identified components: {', '.join(state['requirement_components'])}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state

    async def _plan_implementation(self, state: PrefectAgentState) -> PrefectAgentState:
        """Create an implementation plan for the workflow."""
        print("📝 PLANNING IMPLEMENTATION...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.PLAN_IMPLEMENTATION
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Build context from previous analyses
        component_context = ""
        if 'component_analysis' in state['requirement_analysis']:
            component_context = f"COMPONENT ANALYSIS:\n{state['requirement_analysis']['component_analysis']}"
        
        api_context = ""
        if state['api_info']:
            api_context = "API INFORMATION:\n"
            for name, details in state['api_info'].items():
                # Select key information needed for implementation planning
                api_context += f"\n- {name}:\n"
                # Add authentication method
                auth = details.get('authentication', {})
                if auth:
                    api_context += f"  - Authentication Method: {auth.get('method', 'Unknown')}\n"
                    api_context += f"  - Auth Parameter: {auth.get('parameter', 'Unknown')} in {auth.get('location', 'Unknown')}\n"
                
                # Add key endpoints
                endpoints = details.get('endpoints', [])
                if endpoints:
                    api_context += "  - Key Endpoints:\n"
                    for i, endpoint in enumerate(endpoints[:3]):  # Show top 3 endpoints
                        api_context += f"    * {endpoint.get('name', f'Endpoint {i+1}')}: {endpoint.get('method', 'GET')} {endpoint.get('url', '')}\n"
                
                # Add rate limits for retry planning
                rate_limits = details.get('rate_limits', {})
                if rate_limits:
                    api_context += "  - Rate Limits (important for retry configuration):\n"
                    if isinstance(rate_limits, dict):
                        for limit_type, limit_value in rate_limits.items():
                            api_context += f"    * {limit_type}: {limit_value}\n"
                    else:
                        api_context += f"    * {rate_limits}\n"
        
        # Get relevant documentation
        docs = self._retrieve_relevant_docs(f"Prefect 3.x implementation planning for: {state['requirement']}")
        
        # Prepare the prompt
        system_prompt = self.system_prompts[f"{Phase.ANALYSIS}_{SubState.PLAN_IMPLEMENTATION}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Create a detailed implementation plan for this Prefect workflow:
            
            REQUIREMENTS:
            {state['requirement']}
            
            {component_context}
            
            {api_context}
            
            {f'RELEVANT DOCUMENTATION:\n{docs}' if docs else ''}
            
            Create a step-by-step implementation plan that includes:
            1. Task structure (what discrete tasks are needed)
            2. Task dependencies and data flow
            3. Error handling strategy with specific retry policies
            4. Required environment variables and configuration
            5. Scheduling approach
            6. Implementation order (what to build first)
            
            Be specific about Prefect 3.x features to use for each aspect of the implementation.
            """)
        ]
        
        response = self.llm.invoke(messages)
        implementation_plan = response.content
        
        print("\n📋 IMPLEMENTATION PLAN:")
        print("=" * 50)
        print(implementation_plan)
        print("=" * 50)
        
        # Update state
        state['requirement_analysis']['implementation_plan'] = implementation_plan
        
        # Extract any additional secrets from the implementation plan
        additional_secrets = self._extract_potential_secrets(implementation_plan)
        if additional_secrets:
            if 'potential_secrets' not in state:
                state['potential_secrets'] = []
            new_secrets = [s for s in additional_secrets if s not in state['potential_secrets']]
            if new_secrets:
                print(f"🔑 Additional secrets identified: {', '.join(new_secrets)}")
                state['potential_secrets'].extend(new_secrets)
        
        # Transition to the Generation phase
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.GENERATION
        state['current_substate'] = SubState.GENERATE_SKELETON
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Save checkpoint
        self._save_checkpoint(state)
    
        return state
    async def _generate_skeleton(self, state: PrefectAgentState) -> PrefectAgentState:
        """Generate the skeleton code structure for the workflow."""
        print("🏗️ GENERATING CODE SKELETON...")
        
        # Include implementation plan and API info in context
        implementation_context = state['requirement_analysis'].get('implementation_plan', '')
        
        # Build API context with essential info for code generation
        api_context = ""
        if state['api_info']:
            api_context = "API INFORMATION FOR IMPLEMENTATION:\n"
            for name, details in state['api_info'].items():
                api_context += f"\n- {name}:\n"
                # Add authentication info
                auth = details.get('authentication', {})
                if auth:
                    auth_method = auth.get('method', 'Unknown')
                    auth_param = auth.get('parameter', 'Unknown')
                    auth_location = auth.get('location', 'Unknown')
                    api_context += f"  - Authentication: {auth_method} via {auth_location} parameter '{auth_param}'\n"
                
                # Add key endpoint example
                endpoints = details.get('endpoints', [])
                if endpoints and len(endpoints) > 0:
                    api_context += f"  - Example endpoint: {endpoints[0].get('method', 'GET')} {endpoints[0].get('url', '')}\n"
        
        # Include potential secrets
        secrets_context = ""
        if 'potential_secrets' in state and state['potential_secrets']:
            secrets_context = "ENVIRONMENT VARIABLES TO INCLUDE:\n" + "\n".join(
                [f"- {s}" for s in state['potential_secrets']]
            )
        
        # Get relevant documentation examples
        docs = self._retrieve_relevant_docs(f"Prefect 3.x code skeleton examples for: {state['requirement']}")
        
        # Prepare the prompt
        system_prompt = self.system_prompts[f"{Phase.GENERATION}_{SubState.GENERATE_SKELETON}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Generate a complete, runnable skeleton for this Prefect workflow:
            
            REQUIREMENTS:
            {state['requirement']}
            
            IMPLEMENTATION PLAN:
            {implementation_context}
            
            {api_context}
            
            {secrets_context}
            
            {f'RELEVANT EXAMPLES:\n{docs}' if docs else ''}
            
            Create a full code skeleton with:
            1. All necessary imports
            2. Task and flow function definitions with proper decorators
            3. Docstrings for all functions
            4. Main execution block
            5. Comments indicating where implementation details will go
            6. Proper parameter types and defaults
            
            The skeleton should run without errors, even with placeholder implementations.
            Focus on structure rather than detailed implementation.
            """)
        ]
        
        response = self.llm.invoke(messages)
        skeleton_code = self._extract_code(response.content)
        
        print("\n📄 GENERATED SKELETON:")
        print("=" * 50)
        print(skeleton_code)
        print("=" * 50)
        
        # Update state
        state['code'] = skeleton_code
        state['code_history'] = [skeleton_code]
        
        # Extract secrets from the skeleton
        detected_secrets = self._detect_secrets(skeleton_code)
        if detected_secrets:
            state['secrets'] = detected_secrets
            print(f"🔑 Detected secrets in skeleton: {', '.join(detected_secrets)}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state

    async def _implement_tasks(self, state: PrefectAgentState) -> PrefectAgentState:
        """Implement task functionality in the workflow."""
        print("🔨 IMPLEMENTING TASKS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.IMPLEMENT_TASKS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Get implementation plan and API details for context
        implementation_plan = state['requirement_analysis'].get('implementation_plan', '')
        
        # Build API context with example requests/responses for implementation
        api_context = ""
        if state['api_info']:
            api_context = "API IMPLEMENTATION DETAILS:\n"
            for name, details in state['api_info'].items():
                api_context += f"\n- {name}:\n"
                
                # Add authentication for implementation
                auth = details.get('authentication', {})
                if auth:
                    api_context += f"  - Authentication: {auth.get('method', 'Unknown')} using parameter '{auth.get('parameter', 'Unknown')}' in {auth.get('location', 'Unknown')}\n"
                
                # Add example request code if available
                if details.get('example_request'):
                    api_context += "  - Example request code:\n```python\n"
                    api_context += details.get('example_request') + "\n```\n"
                
                # Add rate limits for retry configuration
                rate_limits = details.get('rate_limits', {})
                if rate_limits:
                    api_context += "  - Configure retries based on these rate limits:\n"
                    if isinstance(rate_limits, dict):
                        for k, v in rate_limits.items():
                            api_context += f"    * {k}: {v}\n"
                    else:
                        api_context += f"    * {rate_limits}\n"
        
        # Get relevant documentation
        docs = self._retrieve_relevant_docs(f"Prefect 3.x task implementation for: {state['requirement']}")
        
        # Prepare the prompt
        system_prompt = self.system_prompts[f"{Phase.GENERATION}_{SubState.IMPLEMENT_TASKS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Implement the task functionality for this Prefect workflow:
            
            CURRENT CODE SKELETON:
            ```python
            {state['code']}
            ```
            
            IMPLEMENTATION PLAN:
            {implementation_plan}
            
            {api_context}
            
            {f'RELEVANT DOCUMENTATION:\n{docs}' if docs else ''}
            
            Focus on implementing the task functions with:
            1. Complete implementation logic for each task
            2. Proper error handling with try/except blocks
            3. Appropriate retry configuration based on the task purpose
            4. Logging at key points using get_run_logger()
            5. Processing of inputs and outputs with proper validation
            
            Keep the flow definition and main execution block as they are for now.
            """)
        ]
        
        response = self.llm.invoke(messages)
        tasks_code = self._extract_code(response.content)
        
        print("\n📄 IMPLEMENTED TASKS:")
        print("=" * 50)
        print(tasks_code)
        print("=" * 50)
        
        # Update state
        state['code'] = tasks_code
        state['code_history'].append(tasks_code)
        
        # Extract any new secrets
        new_secrets = self._detect_secrets(tasks_code)
        if new_secrets:
            if 'secrets' not in state:
                state['secrets'] = []
            state['secrets'] = list(set(state['secrets'] + new_secrets))
            
            # Identify any new secrets not previously detected
            prev_secrets = set(state.get('secrets', []))
            added_secrets = [s for s in new_secrets if s not in prev_secrets]
            if added_secrets:
                print(f"🔑 New secrets detected: {', '.join(added_secrets)}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _implement_flows(self, state: PrefectAgentState) -> PrefectAgentState:
        """Implement flow orchestration logic."""
        print("🔄 IMPLEMENTING FLOWS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.IMPLEMENT_FLOWS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Get implementation plan
        implementation_plan = state['requirement_analysis'].get('implementation_plan', '')
        
        # Get relevant documentation
        docs = self._retrieve_relevant_docs(f"Prefect 3.x flow implementation for: {state['requirement']}")
        
        # Prepare the prompt
        system_prompt = self.system_prompts[f"{Phase.GENERATION}_{SubState.IMPLEMENT_FLOWS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Implement the flow orchestration logic for this Prefect workflow:
            
            CURRENT CODE WITH TASKS:
            ```python
            {state['code']}
            ```
            
            IMPLEMENTATION PLAN:
            {implementation_plan}
            
            {f'RELEVANT DOCUMENTATION:\n{docs}' if docs else ''}
            
            Focus on implementing the flow logic with:
            1. Proper task orchestration and dependencies
            2. Parameter passing between tasks
            3. Conditional branching if needed
            4. Result handling and aggregation
            5. Flow-level logging
            
            Ensure the flow properly orchestrates all the tasks and maintains the correct data flow.
            """)
        ]
        
        response = self.llm.invoke(messages)
        flows_code = self._extract_code(response.content)
        
        print("\n📄 IMPLEMENTED FLOWS:")
        print("=" * 50)
        print(flows_code)
        print("=" * 50)
        
        # Update state
        state['code'] = flows_code
        state['code_history'].append(flows_code)
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _add_error_handling(self, state: PrefectAgentState) -> PrefectAgentState:
        """Add comprehensive error handling to the workflow."""
        print("🛡️ ADDING ERROR HANDLING...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.ADD_ERROR_HANDLING
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Extract API rate limits for retry configuration
        api_retry_context = ""
        if state['api_info']:
            api_retry_context = "API RATE LIMITS FOR RETRY CONFIGURATION:\n"
            for name, details in state['api_info'].items():
                rate_limits = details.get('rate_limits', {})
                if rate_limits:
                    api_retry_context += f"\n- {name}:\n"
                    if isinstance(rate_limits, dict):
                        for k, v in rate_limits.items():
                            api_retry_context += f"  - {k}: {v}\n"
                    else:
                        api_retry_context += f"  - {rate_limits}\n"
        
        # Get relevant documentation
        docs = self._retrieve_relevant_docs(f"Prefect 3.x error handling best practices")
        
        # Prepare the prompt
        system_prompt = self.system_prompts[f"{Phase.GENERATION}_{SubState.ADD_ERROR_HANDLING}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Enhance the workflow with comprehensive error handling:
            
            CURRENT WORKFLOW CODE:
            ```python
            {state['code']}
            ```
            
            {api_retry_context}
            
            {f'ERROR HANDLING PATTERNS:\n{docs}' if docs else ''}
            
            Implement robust error handling including:
            1. Task-level try/except blocks with specific exception types
            2. Retry policies with exponential backoff for external integrations
            3. Fallback strategies for critical operations
            4. Detailed error logging for troubleshooting
            5. Flow-level error handling for graceful failure
            
            Make sure the workflow is resilient against common failure modes.
            """)
        ]
        
        response = self.llm.invoke(messages)
        error_handling_code = self._extract_code(response.content)
        
        print("\n📄 ADDED ERROR HANDLING:")
        print("=" * 50)
        print(error_handling_code)
        print("=" * 50)
        
        # Update state
        state['code'] = error_handling_code
        state['code_history'].append(error_handling_code)
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _add_scheduling(self, state: PrefectAgentState) -> PrefectAgentState:
        """Add scheduling capabilities to the workflow."""
        print("⏰ ADDING SCHEDULING...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.ADD_SCHEDULING
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Extract any scheduling requirements from the initial requirements
        requirement = state['requirement']
        scheduling_requirements = ""
        
        # Look for scheduling patterns in the requirement
        schedule_patterns = [
            r'(?:every|each)\s+(\d+)\s+(minutes?|hours?|days?|weeks?|months?)',
            r'(?:daily|weekly|monthly|hourly)',
            r'(?:at|on)\s+(\d{1,2}(?::\d{2})?(?:\s*[ap]m)?)',
            r'(?:every|each)\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)',
            r'([0-9*]+\s+[0-9*]+\s+[0-9*]+\s+[0-9*]+\s+[0-9*]+)'  # cron expression
        ]
        
        for pattern in schedule_patterns:
            matches = re.findall(pattern, requirement, re.IGNORECASE)
            if matches:
                scheduling_requirements = "Explicit scheduling requirement found in requirements."
                break
        
        # Get relevant documentation for scheduling
        docs = self._retrieve_relevant_docs(f"Prefect 3.x scheduling and deployments")
        
        # Prepare the prompt
        system_prompt = self.system_prompts[f"{Phase.GENERATION}_{SubState.ADD_SCHEDULING}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Add scheduling capabilities to this Prefect workflow:
            
            CURRENT WORKFLOW CODE:
            ```python
            {state['code']}
            ```
            
            ORIGINAL REQUIREMENTS:
            {state['requirement']}
            
            {scheduling_requirements}
            
            {f'SCHEDULING PATTERNS:\n{docs}' if docs else ''}
            
            Implement appropriate scheduling:
            1. Set up scheduling based on the requirements
            2. Use Prefect 3.x deployment functionality for scheduling
            3. Include appropriate timezone handling if needed
            4. Ensure the main block supports both direct execution and scheduled runs
            5. Document how the scheduling works
            
            Make sure the scheduling implementation uses current Prefect 3.x patterns.
            """)
        ]
        
        response = self.llm.invoke(messages)
        scheduling_code = self._extract_code(response.content)
        
        print("\n📄 ADDED SCHEDULING:")
        print("=" * 50)
        print(scheduling_code)
        print("=" * 50)
        
        # Update state
        state['code'] = scheduling_code
        state['code_history'].append(scheduling_code)
        
        # Transition to validation phase
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_phase'] = Phase.VALIDATION
        state['current_substate'] = SubState.VALIDATE_SYNTAX
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _validate_syntax(self, state: PrefectAgentState) -> PrefectAgentState:
        """Validate the syntax of the generated code."""
        print("🔍 VALIDATING SYNTAX...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.VALIDATE_SYNTAX
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Check for syntax errors using Python's compile function
        syntax_valid = True
        error_message = None
        
        try:
            compile(state['code'], "<string>", "exec")
        except SyntaxError as e:
            syntax_valid = False
            error_message = f"Syntax error: {str(e)}"
            
            # Create structured error info
            error_info = ErrorInfo(
                message=str(e),
                error_type="SyntaxError",
                line_number=e.lineno,
                code_snippet=e.text.strip() if hasattr(e, 'text') and e.text else None,
                source="compilation"
            )
            
            # Add to errors list
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(error_info)
        
        if syntax_valid:
            print("✅ SYNTAX VALIDATION PASSED")
        else:
            print(f"❌ SYNTAX VALIDATION FAILED: {error_message}")
            
            # Use LLM to suggest a fix
            system_prompt = self.system_prompts[f"{Phase.VALIDATION}_{SubState.VALIDATE_SYNTAX}"]
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
                Fix the syntax error in this Prefect workflow code:
                
                CODE:
                ```python
                {state['code']}
                ```
                
                ERROR:
                {error_message}
                
                Please provide a detailed analysis of the syntax error and the complete corrected code.
                """)
            ]
            
            response = self.llm.invoke(messages)
            analysis = response.content
            
            # Extract the fixed code
            fixed_code = self._extract_code(response.content)
            
            print("\n🔧 SYNTAX ERROR ANALYSIS:")
            print("=" * 50)
            print(analysis)
            print("=" * 50)
            
            if fixed_code:
                print("\n📄 SYNTAX FIXED CODE:")
                print("=" * 50)
                print(fixed_code)
                print("=" * 50)
                
                # Update state with fixed code
                state['code'] = fixed_code
                state['code_history'].append(fixed_code)
                
                # Update the error to mark it as fixed
                if 'errors' in state and state['errors']:
                    state['errors'][-1].is_fixed = True
                    state['errors'][-1].attempted_fixes.append("Syntax error fixed with LLM assistance")
            
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _validate_imports(self, state: PrefectAgentState) -> PrefectAgentState:
        """Validate the imports in the generated code."""
        print("📦 VALIDATING IMPORTS...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.VALIDATE_IMPORTS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Extract all import statements
        import_lines = []
        for line in state['code'].split('\n'):
            if line.strip().startswith(('import ', 'from ')):
                import_lines.append(line.strip())
        
        # LLM check for import issues
        system_prompt = self.system_prompts[f"{Phase.VALIDATION}_{SubState.VALIDATE_IMPORTS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Validate the imports in this Prefect workflow code:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            IDENTIFIED IMPORT STATEMENTS:
            {chr(10).join(import_lines)}
            
            Check for:
            1. Missing imports that are used in the code
            2. Incorrect import syntax
            3. Imports that don't match Prefect 3.x patterns
            4. Unused imports
            5. Dependencies that would need to be installed
            
            Identify any issues and suggest fixes. If everything looks correct, confirm that.
            """)
        ]
        
        response = self.llm.invoke(messages)
        import_analysis = response.content
        
        print("\n📋 IMPORT ANALYSIS:")
        print("=" * 50)
        print(import_analysis)
        print("=" * 50)
        
        # Check if there are import issues identified
        import_issues = False
        if any(phrase in import_analysis.lower() for phrase in [
            "missing import", "incorrect import", "unused import", 
            "need to install", "should be imported", "not imported", 
            "import error", "dependency", "package not found"
        ]):
            import_issues = True
        
        if import_issues:
            print("⚠️ IMPORT ISSUES DETECTED")
            
            # Create structured error info
            error_info = ErrorInfo(
                message="Import validation issues detected",
                error_type="ImportError",
                source="validation",
                code_snippet="\n".join(import_lines)
            )
            
            # Add to errors list
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(error_info)
            
            # Get fixed code from LLM
            messages = [
                SystemMessage(content=self.system_prompts[f"{Phase.VALIDATION}_{SubState.FIX_ERRORS}"]),
                HumanMessage(content=f"""
                Fix the import issues in this Prefect workflow code:
                
                CODE:
                ```python
                {state['code']}
                ```
                
                IMPORT ISSUES ANALYSIS:
                {import_analysis}
                
                Please provide the complete fixed code with corrected imports.
                Make sure all necessary imports are included and properly formatted.
                """)
            ]
            
            response = self.llm.invoke(messages)
            fixed_code = self._extract_code(response.content)
            
            if fixed_code:
                print("\n📄 IMPORT FIXED CODE:")
                print("=" * 50)
                print(fixed_code)
                print("=" * 50)
                
                # Update state with fixed code
                state['code'] = fixed_code
                state['code_history'].append(fixed_code)
                
                # Update the error to mark it as fixed
                if 'errors' in state and state['errors']:
                    state['errors'][-1].is_fixed = True
                    state['errors'][-1].attempted_fixes.append("Import issues fixed with LLM assistance")
        else:
            print("✅ IMPORT VALIDATION PASSED")
        
        # Extract required packages for installation instructions
        required_packages = set(["prefect>=2.0.0"])  # Always include Prefect
        
        for line in import_lines:
            # Extract the package name from import statements
            if line.startswith('import '):
                package = line.split('import ')[1].split(' as ')[0].split(',')[0].strip()
                base_package = package.split('.')[0]
                if base_package not in ['os', 'sys', 'datetime', 're', 'json', 'time', 'pathlib', 'typing']:
                    required_packages.add(base_package)
            elif line.startswith('from '):
                package = line.split('from ')[1].split(' import')[0].strip()
                base_package = package.split('.')[0]
                if base_package not in ['os', 'sys', 'datetime', 're', 'json', 'time', 'pathlib', 'typing']:
                    required_packages.add(base_package)
        
        # Store required packages in state
        state['required_packages'] = list(required_packages)
        print(f"📦 Required packages: {', '.join(required_packages)}")
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
    
    async def _validate_execution(self, state: PrefectAgentState) -> PrefectAgentState:
        """Validate that the code can be executed."""
        print("🧪 VALIDATING EXECUTION...")
        
        # Update state
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.VALIDATE_EXECUTION
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Attempt to validate execution using the virtual environment
        is_valid, error = self._validate_code_in_venv(state['code'])
        
        if is_valid:
            print("✅ EXECUTION VALIDATION PASSED")
        else:
            print(f"❌ EXECUTION VALIDATION FAILED: {error}")
            
            # Create structured error info
            error_info = ErrorInfo(
                message=error,
                error_type="ExecutionError",
                source="execution"
            )
            
            # Add to errors list
            if 'errors' not in state:
                state['errors'] = []
            state['errors'].append(error_info)
            
            # Set current substate to CLASSIFY_ERRORS to handle the error
            state['previous_phase'] = state['current_phase']
            state['previous_substate'] = state['current_substate']
            state['current_substate'] = SubState.CLASSIFY_ERRORS
            state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _classify_errors(self, state: PrefectAgentState) -> PrefectAgentState:
        """Classify errors to guide the fixing process."""
        print("🔍 CLASSIFYING ERRORS...")
        
        # Skip if no errors
        if 'errors' not in state or not state['errors']:
            print("No errors to classify.")
            return state
        
        # Get the most recent error
        latest_error = state['errors'][-1]
        
        # Prepare context with error details
        error_context = f"""
        ERROR MESSAGE: {latest_error.message}
        ERROR TYPE: {latest_error.error_type}
        SOURCE: {latest_error.source}
        """
        
        if latest_error.line_number:
            error_context += f"LINE NUMBER: {latest_error.line_number}\n"
            
        if latest_error.code_snippet:
            error_context += f"CODE SNIPPET: {latest_error.code_snippet}\n"
        
        # Use LLM to classify the error
        system_prompt = self.system_prompts[f"{Phase.VALIDATION}_{SubState.CLASSIFY_ERRORS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Classify this error in the Prefect workflow code:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            ERROR DETAILS:
            {error_context}
            
            Provide a detailed classification of this error including:
            1. Root cause category (syntax, logic, dependency, configuration, etc.)
            2. Severity (critical, high, medium, low)
            3. Affected component (which part of the code is problematic)
            4. Potential fixes
            5. Whether this is a Prefect-specific issue or a general Python issue
            """)
        ]
        
        response = self.llm.invoke(messages)
        error_classification = response.content
        
        print("\n📋 ERROR CLASSIFICATION:")
        print("=" * 50)
        print(error_classification)
        print("=" * 50)
        
        # Store the classification
        latest_error.classification = error_classification
        
        # Update substate to fix errors
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.FIX_ERRORS
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _fix_errors(self, state: PrefectAgentState) -> PrefectAgentState:
        """Fix identified errors in the code."""
        print("🔧 FIXING ERRORS...")
        
        # Skip if no errors
        if 'errors' not in state or not state['errors'] or all(e.is_fixed for e in state['errors']):
            print("No errors to fix.")
            
            # If we've gotten here with no errors to fix, move to finalization
            state['previous_phase'] = state['current_phase']
            state['previous_substate'] = state['current_substate']
            state['current_phase'] = Phase.FINALIZATION
            state['current_substate'] = SubState.OPTIMIZE_CODE
            state['phase_history'].append((state['current_phase'], state['current_substate']))
            
            return state
        
        # Get the unfixed errors
        unfixed_errors = [e for e in state['errors'] if not e.is_fixed]
        latest_error = unfixed_errors[-1]
        
        # Get error classification if available
        error_context = f"""
        ERROR MESSAGE: {latest_error.message}
        ERROR TYPE: {latest_error.error_type}
        SOURCE: {latest_error.source}
        """
        
        if hasattr(latest_error, 'classification'):
            error_context += f"\nCLASSIFICATION:\n{latest_error.classification}\n"
        
        # Use LLM to fix the error
        system_prompt = self.system_prompts[f"{Phase.VALIDATION}_{SubState.FIX_ERRORS}"]
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"""
            Fix this error in the Prefect workflow code:
            
            CODE:
            ```python
            {state['code']}
            ```
            
            ERROR DETAILS:
            {error_context}
            
            Please provide:
            1. A detailed analysis of what's causing the error
            2. The complete fixed code that resolves the issue
            
            Your fix should address the root cause, not just mask the symptoms.
            """)
        ]
        
        response = self.llm.invoke(messages)
        fix_analysis = response.content
        
        # Extract fixed code
        fixed_code = self._extract_code(fix_analysis)
        
        print("\n📋 ERROR FIX ANALYSIS:")
        print("=" * 50)
        print(fix_analysis)
        print("=" * 50)
        
        if fixed_code:
            print("\n📄 FIXED CODE:")
            print("=" * 50)
            print(fixed_code)
            print("=" * 50)
            
            # Update state with fixed code
            state['code'] = fixed_code
            state['code_history'].append(fixed_code)
            
            # Update the error to mark it as fixed
            latest_error.is_fixed = True
            latest_error.attempted_fixes.append("Error fixed with LLM assistance")
            
            # Return to validation
            state['previous_phase'] = state['current_phase']
            state['previous_substate'] = state['current_substate']
            state['current_substate'] = SubState.VALIDATE_EXECUTION
            state['phase_history'].append((state['current_phase'], state['current_substate']))
        else:
            print("⚠️ Could not extract fixed code. Moving to try a different approach.")
            
            # Try a different approach if we couldn't fix the error
            await self._try_different_approach(state)
        
        # Save checkpoint
        self._save_checkpoint(state)
        
        return state
        
    async def _try_different_approach(self, state: PrefectAgentState) -> PrefectAgentState:
        """Try a completely different approach when stuck in a loop."""
        print("🔄 TRYING A DIFFERENT APPROACH...")
        
        # Get recent errors
        recent_errors = state['errors'][-3:] if len(state['errors']) >= 3 else state['errors']
        error_patterns = "\n".join([f"- {err.message}" for err in recent_errors])
        
        # Get API information if available
        api_context = ""
        if state['api_info']:
            api_context = "API INFORMATION TO CONSIDER:\n"
            for name, details in state['api_info'].items():
                # Extract key details
                auth = details.get('authentication', {})
                if auth:
                    auth_method = auth.get('method', 'Unknown')
                    auth_param = auth.get('parameter', 'Unknown')
                    auth_location = auth.get('location', 'Unknown')
                    api_context += f"- {name} requires {auth_method} authentication via {auth_location} parameter '{auth_param}'\n"
                
                # Add a sample endpoint
                endpoints = details.get('endpoints', [])
                if endpoints and len(endpoints) > 0:
                    sample_endpoint = endpoints[0]
                    api_context += f"- Primary endpoint: {sample_endpoint.get('name', 'Unknown')} - {sample_endpoint.get('url', '')}\n"
                
                # Add rate limit info if present
                rate_limits = details.get('rate_limits', {})
                if rate_limits:
                    api_context += f"- {name} has rate limits that should be handled with retries\n"
        
        # Get any detected secrets
        secrets_context = ""
        if 'secrets' in state and state['secrets']:
            secrets_context = "ENVIRONMENT VARIABLES TO MAINTAIN:\n" + "\n".join([f"- {s}" for s in state['secrets']])
            secrets_context += "\nEnsure these are properly handled in the new implementation."
        
        # Get relevant example documentation
        example_docs = self._retrieve_relevant_docs(f"Prefect 3.x examples for {state['requirement']}")
        
        if example_docs:
            print("📝 Found relevant example documentation")
            example_context = f"""
            RELEVANT PREFECT EXAMPLES:
            {example_docs}
            
            Consider these examples for a different implementation approach.
            """
        else:
            example_context = ""
        
        messages = [
            SystemMessage(content=self.system_prompts["base"]),
            HumanMessage(content=f"""
            We've been stuck in a loop trying to fix this Prefect workflow.
            Let's try a completely different approach.
            
            ORIGINAL REQUIREMENT:
            {state['requirement']}
            
            RECENT ERRORS ENCOUNTERED:
            {error_patterns}
            
            {api_context}
            
            {example_context}
            
            {secrets_context}
            
            Here's a template for a working Prefect 3.x workflow to guide you:
            
            ```python
            import os
            from datetime import datetime, timedelta
            from prefect import flow, task, get_run_logger
            
            @task(retries=3, retry_delay_seconds=30)
            def sample_task(param):
                logger = get_run_logger()
                logger.info(f"Processing {param}")
                return param
            
            @flow(name="Sample Flow")
            def sample_flow():
                logger = get_run_logger()
                logger.info("Starting flow")
                result = sample_task("sample_data")
                logger.info(f"Flow completed with result: {result}")
            
            if __name__ == "__main__":
                # For local execution
                sample_flow()
            ```
            
            Please generate a NEW implementation using a different approach:
            1. Use a simpler structure
            2. Avoid the patterns that led to previous errors
            3. Focus on core functionality first
            4. Use the most basic and reliable Prefect 3.x patterns
            
            Provide complete, executable Python code for a Prefect 3.x workflow.
            """)
        ]
        
        response = self.llm.invoke(messages)
        new_code = self._extract_code(response.content)
        
        print("\n📄 NEW APPROACH CODE:")
        print("=" * 50)
        print(new_code)
        print("=" * 50)
        
        # Update state
        state['code'] = new_code
        state['code_history'].append(new_code)
        
        # Detect any new secrets
        new_secrets = self._detect_secrets(new_code)
        if new_secrets:
            if 'secrets' not in state:
                state['secrets'] = []
            state['secrets'] = list(set(state['secrets'] + new_secrets))
        
        # Return to validation
        state['previous_phase'] = state['current_phase']
        state['previous_substate'] = state['current_substate']
        state['current_substate'] = SubState.VALIDATE_SYNTAX
        state['phase_history'].append((state['current_phase'], state['current_substate']))
        
        # Save checkpoint
        self._save_checkpoint(state)
    
        return state
    
    def _validate_code_in_venv(self, code: str) -> Tuple[bool, Optional[str]]:
        """Validate the generated code in a clean virtual environment."""
        # First check for basic syntax errors
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        
        # Create temporary directory and files
        temp_dir = Path(tempfile.mkdtemp())
        code_file = temp_dir / "workflow.py"
        
        # Write the code to a file
        with open(code_file, "w") as f:
            f.write(code)
        
        try:
            # Create virtual environment
            env_dir, python_path = self._create_virtual_env(temp_dir / "venv")
            
            # Check if the code has required environment variables
            env_vars = self._detect_secrets(code)
            
            # Set environment variables if needed
            env = os.environ.copy()
            if env_vars:
                print("\n⚠️ This workflow requires the following environment variables:")
                for var in env_vars:
                    print(f"  - {var}")
                
                # Create a temporary .env file template
                with open(temp_dir / ".env.temp", "w") as f:
                    for var in env_vars:
                        f.write(f"{var}=\n")
                
                print("\nA template .env.temp file has been created.")
                
                # In non-interactive mode, just set placeholder values
                for var in env_vars:
                    env[var] = f"placeholder_{var.lower()}_value"
            
            # Detect needed packages from imports
            needed_packages = ["prefect>=2.0.0", "pandas", "requests"]
            if "import apscheduler" in code or "from apscheduler" in code:
                needed_packages.append("apscheduler")
            
            # Install dependencies
            print(f"📦 Installing packages: {', '.join(needed_packages)}...")
            if not self._install_in_venv(python_path, needed_packages):
                return False, "Failed to install required packages in virtual environment"
            
            # Create a simplified version of the code if it's using deployment functionality
            if "deployment.apply()" in code or "Deployment." in code:
                # Create a simplified version that just runs the flow
                simplified_code = code
                # Find the flow function name
                flow_match = re.search(r'@flow\s+def\s+(\w+)', code)
                if flow_match:
                    flow_name = flow_match.group(1)
                    # Replace the deployment code with direct flow execution
                    simplified_code = re.sub(
                        r'if\s+__name__\s*==\s*"__main__".*',
                        f'if __name__ == "__main__":\n    # Running flow directly instead of using deployment\n    {flow_name}()',
                        simplified_code, 
                        flags=re.DOTALL
                    )
                    # Write the simplified code to a separate file
                    simple_file = temp_dir / "simplified_workflow.py"
                    with open(simple_file, "w") as f:
                        f.write(simplified_code)
                    
                    print("\n🔍 Executing simplified workflow to validate flow structure...")
                    simple_result = subprocess.run(
                        [str(python_path), simple_file],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        env=env
                    )
                    
                    if simple_result.returncode == 0:
                        print("✅ Simplified workflow executed successfully!")
                        # If the simplified version works, we can consider the validation successful
                        return True, None
                    else:
                        print(f"❌ Even simplified workflow failed: {simple_result.stderr}")
            
            # Run the original workflow with a timeout to prevent hanging
            print("\n🔍 Executing workflow in virtual environment to validate...")
            try:
                result = subprocess.run(
                    [str(python_path), code_file],
                    capture_output=True,
                    text=True,
                    timeout=60,  # 1 minute timeout
                    env=env
                )
                
                if result.returncode != 0:
                    print(f"❌ Workflow execution failed with error:")
                    print(result.stderr)
                    return False, f"Workflow execution failed: {result.stderr}"
                
                print("✅ Workflow executed successfully in virtual environment!")
                return True, None
                
            except subprocess.TimeoutExpired:
                print("⚠️ Workflow execution timed out. This might be normal for long-running workflows.")
                return True, "Validation timed out, but code appears valid"
            except Exception as e:
                print(f"❌ Exception during validation: {str(e)}")
                return False, f"Validation error: {str(e)}"
        finally:
            # Clean up
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"Warning: Failed to clean up temporary directory: {str(e)}")

    def _create_virtual_env(self, env_dir=None):
        """Create a virtual environment for validation."""
        if env_dir is None:
            # Create a temporary directory for the virtual environment
            env_dir = Path(tempfile.mkdtemp()) / "venv"
        
        print(f"\n🔧 Creating virtual environment at {env_dir}...")
        venv.create(env_dir, with_pip=True)
        
        # Get the path to the Python executable in the virtual environment
        if os.name == 'nt':  # Windows
            python_path = env_dir / "Scripts" / "python.exe"
        else:  # Unix/Linux/Mac
            python_path = env_dir / "bin" / "python"
        
        return env_dir, python_path

    def _install_in_venv(self, python_path, packages):
        """Install packages in the virtual environment."""
        print(f"📦 Installing packages: {', '.join(packages)}...")
        try:
            subprocess.run(
                [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
                check=True, capture_output=True, text=True
            )
            
            # Install the packages
            subprocess.run(
                [str(python_path), "-m", "pip", "install"] + packages,
                check=True, capture_output=True, text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e.stderr}")
            return False

    def _should_request_clarification(self, state: PrefectAgentState) -> bool:
        """Determine if clarification should be requested from the user."""
        # Skip clarification in non-interactive mode
        if not self.interactive:
            return False
        
        # Don't ask for clarification too frequently
        if 'conversation' in state and state['conversation'].clarification_count >= 3:
            return False
        
        # Potentially request clarification in these cases:
        
        # 1. During API analysis if API details are sparse
        if (state['current_phase'] == Phase.ANALYSIS and 
            state['current_substate'] == SubState.EXTRACT_APIS and
            'api_info' in state and state['api_info']):
            
            # Check if any API has minimal information
            for api_name, api_details in state['api_info'].items():
                if not api_details.get('endpoints') or not api_details.get('authentication'):
                    return True
        
        # 2. After encountering persistent errors
        if (state['current_phase'] == Phase.VALIDATION and
            'errors' in state and len(state['errors']) >= 2):
            return True
        
        # 3. When planning implementation with complex requirements
        if (state['current_phase'] == Phase.ANALYSIS and
            state['current_substate'] == SubState.PLAN_IMPLEMENTATION and
            len(state['requirement'].split()) > 100):  # Long requirements
            return True
        
        return False

    def _build_graph(self) -> StateGraph:
        """Build the FSM graph for the workflow generation process."""
        # Define the state graph
        builder = StateGraph(PrefectAgentState)
        
        # Add nodes for all phases
        
        # Analysis Phase
        builder.add_node("initialize", self._initialize_phase)
        builder.add_node("parse_requirements", self._parse_requirements)
        builder.add_node("extract_apis", self._extract_apis)
        builder.add_node("analyze_components", self._analyze_components)
        builder.add_node("plan_implementation", self._plan_implementation)
        
        # Generation Phase
        builder.add_node("generate_skeleton", self._generate_skeleton)
        builder.add_node("implement_tasks", self._implement_tasks)
        builder.add_node("implement_flows", self._implement_flows)
        builder.add_node("add_error_handling", self._add_error_handling)
        builder.add_node("add_scheduling", self._add_scheduling)
        
        # Validation Phase
        builder.add_node("validate_syntax", self._validate_syntax)
        builder.add_node("validate_imports", self._validate_imports)
        builder.add_node("validate_execution", self._validate_execution)
        builder.add_node("classify_errors", self._classify_errors)
        builder.add_node("fix_errors", self._fix_errors)
        builder.add_node("try_different_approach", self._try_different_approach)
        
        # Finalization Phase
        builder.add_node("optimize_code", self._optimize_code)
        builder.add_node("generate_documentation", self._generate_documentation)
        builder.add_node("finalize_output", self._finalize_output)
        builder.add_node("package_deployment", self._package_deployment)
        builder.add_node("generate_tests", self._generate_tests)
        
        # Conversation Layer
        builder.add_node("request_clarification", self._request_clarification)
        builder.add_node("process_feedback", self._process_feedback)
        builder.add_node("explain_code", self._explain_code)
        builder.add_node("suggest_improvements", self._suggest_improvements)
        builder.add_node("demo_usage", self._demo_usage)
        builder.add_node("checkpoint_management", self._checkpoint_management)
        
        # Define the edges for normal flow
        
        # Analysis Phase Flow
        builder.add_edge("initialize", "parse_requirements")
        builder.add_edge("parse_requirements", "extract_apis")
        builder.add_edge("extract_apis", "analyze_components")
        builder.add_edge("analyze_components", "plan_implementation")
        builder.add_edge("plan_implementation", "generate_skeleton")
        
        # Generation Phase Flow
        builder.add_edge("generate_skeleton", "implement_tasks")
        builder.add_edge("implement_tasks", "implement_flows")
        builder.add_edge("implement_flows", "add_error_handling")
        builder.add_edge("add_error_handling", "add_scheduling")
        builder.add_edge("add_scheduling", "validate_syntax")
        
        # Validation Phase Flow
        builder.add_edge("validate_syntax", "validate_imports")
        builder.add_edge("validate_imports", "validate_execution")
        builder.add_edge("classify_errors", "fix_errors")
        builder.add_edge("fix_errors", "validate_execution")  # Loop back for validation
        builder.add_edge("try_different_approach", "validate_syntax")  # After trying different approach
        
        # Conditional transition from validation to finalization
        builder.add_conditional_edges(
            "validate_execution",
            lambda state: "optimize_code" if state.get('success', False) else "classify_errors",
            {
                "optimize_code": "optimize_code",
                "classify_errors": "classify_errors"
            }
        )
        
        # Finalization Phase Flow
        builder.add_edge("optimize_code", "generate_documentation")
        builder.add_edge("generate_documentation", "finalize_output")
        builder.add_edge("finalize_output", "package_deployment")
        builder.add_edge("package_deployment", "generate_tests")
        builder.add_edge("generate_tests", END)
        
        # Conversation Layer - conditional connections
        # These could be triggered at various points depending on state
        
        # Set the entry point
        builder.set_entry_point("initialize")
        
        # Compile the graph
        return builder.compile()

    async def generate_workflow(
        self, 
        requirement: str, 
        model: str = "gpt-4o", 
        max_iterations: int = 8,
        interactive: bool = False,
        api_configs: Dict[str, Dict[str, str]] = None
    ) -> Tuple[str, List[str], bool]:
        """
        Generate a Prefect workflow based on requirements using FSM architecture.
        
        Args:
            requirement: The workflow requirements
            model: The LLM model to use
            max_iterations: Maximum iterations before giving up
            interactive: Whether to enable interactive mode with user feedback
            api_configs: Optional configurations for APIs mentioned in requirements
                Example: {"Alpha Vantage API": {"tier": "free"}}
                
        Returns:
            Tuple of (generated_code, required_secrets, success)
        """
        print(f"🚀 GENERATING PREFECT WORKFLOW")
        print(f"📋 REQUIREMENT:\n{requirement}")
        
        # Update model if needed
        if model != "gpt-4o":
            self.llm = ChatOpenAI(model=model, temperature=0.2)
        
        # Set interactive mode
        self.interactive = interactive
        
        # Initialize state
        initial_state = {
            "requirement": requirement,
            "current_phase": None,
            "current_substate": None,
            "previous_phase": None,
            "previous_substate": None,
            "phase_history": [],
            "code": "",
            "code_history": [],
            "errors": [],
            "reasoning": [],
            "secrets": [],
            "iterations": 0,
            "max_iterations": max_iterations,
            "api_configs": api_configs or {},
            "api_info": {},
            "conversation": ConversationContext(),
            "checkpoints": [],
            "requirement_analysis": {},
            "final_code": "",
            "success": False,
            "message": ""
        }
        
        # Run the graph
        final_state = await self.workflow_graph.ainvoke(initial_state)
        
        # Get the final code and secrets
        final_code = final_state.get('final_code', '') or final_state['code']
        secrets = final_state.get('secrets', [])
        
        # Print summary
        print(f"\n{'✅ SUCCESS' if final_state['success'] else '⚠️ PARTIAL SUCCESS'}: {final_state['message']}")
        
        return final_code, secrets, final_state.get('success', False)
    
    def has_checkpoints(self) -> bool:
        """Check if there are any saved checkpoints."""
        return any(self.checkpoint_dir.glob("*.pkl"))
    
    def list_checkpoints(self) -> List[Checkpoint]:
        """List all available checkpoints."""
        checkpoints = []
        for pkl_file in self.checkpoint_dir.glob("*.pkl"):
            try:
                with open(pkl_file, "rb") as f:
                    checkpoint = pickle.load(f)
                    checkpoints.append(checkpoint)
            except Exception as e:
                print(f"Error loading checkpoint {pkl_file}: {str(e)}")
        
        # Sort by timestamp
        return sorted(checkpoints, key=lambda x: x.timestamp)
    
    def restore_workflow_from_checkpoint(self, checkpoint_id: str) -> Optional[str]:
        """Restore and return workflow code from a specific checkpoint."""
        checkpoint = self._load_checkpoint(checkpoint_id)
        if checkpoint:
            return checkpoint.code
        return None


async def setup_prefect_docs_vectorstore():
    """Connect to an existing Pinecone vector database with Prefect documentation."""
    if not VECTORSTORE_AVAILABLE:
        print("⚠️ Pinecone not available. Cannot setup documentation vectorstore.")
        return None
        
    # Initialize Pinecone
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENVIRONMENT")
    )
    
    # Get index name from environment variables or use defaults
    index_name = os.environ.get("PINECONE_INDEX_NAME", "prefect-docs")
    
    # Connect to the index with OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Return the vectorstore connection
    return PineconeVectorStore(
        index_name=index_name, 
        embedding=embeddings
    )


async def generate_prefect_workflow(
    requirement: str, 
    model: str = "gpt-4o", 
    max_iterations: int = 8,
    interactive: bool = False,
    use_docs: bool = True,
    use_search: bool = False,
    api_configs: Dict[str, Dict[str, str]] = None,
    checkpoint_dir: str = None
):
    """Generate a Prefect workflow using the enhanced FSM agent."""
    # Check for Pinecone credentials if docs enabled
    if use_docs and VECTORSTORE_AVAILABLE:
        if not os.environ.get("PINECONE_API_KEY") or not os.environ.get("PINECONE_ENVIRONMENT"):
            print("⚠️ Warning: PINECONE_API_KEY or PINECONE_ENVIRONMENT not set.")
            print("Documentation retrieval will be disabled.")
            use_docs = False
    elif use_docs:
        print("⚠️ Warning: Pinecone package not available. Documentation retrieval will be disabled.")
        use_docs = False
    
    # Check for Tavily API key if search is enabled
    if use_search and not os.environ.get("TAVILY_API_KEY"):
        print("⚠️ Warning: TAVILY_API_KEY not set.")
        print("API search will be disabled.")
        use_search = False
    
    # Initialize agent
    agent = PrefectWorkflowAgent(
        model_name=model, 
        use_docs=use_docs, 
        use_search=use_search,
        checkpoint_dir=checkpoint_dir
    )
    
    # Generate workflow
    code, secrets, success = await agent.generate_workflow(
        requirement, 
        model=model, 
        max_iterations=max_iterations, 
        interactive=interactive,
        api_configs=api_configs
    )
    
    if success:
        print("✅ Successfully generated a valid Prefect workflow!")
    else:
        print("⚠️ Generated a workflow but it may need manual adjustments.")
    
    return code, secrets, success


if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate Prefect 3.x workflows using an enhanced FSM agent")
    parser.add_argument("--requirement", "-r", type=str, 
                       help="Workflow requirements (text or file path)")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o", 
                       help="LLM model to use (default: gpt-4o)")
    parser.add_argument("--iterations", "-i", type=int, default=8, 
                       help="Maximum iterations (default: 8)")
    parser.add_argument("--interactive", action="store_true", 
                       help="Enable interactive mode with clarification requests")
    parser.add_argument("--no-docs", action="store_true", 
                       help="Disable documentation retrieval")
    parser.add_argument("--use-search", action="store_true", 
                       help="Enable web search for API documentation")
    parser.add_argument("--api-tier", type=str, 
                       help="Specify API tier (e.g., 'free', 'premium')")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory for storing checkpoints")
    
    args = parser.parse_args()
    
    # Get the requirement
    if args.requirement:
        # Check if the input is a file path
        if os.path.isfile(args.requirement):
            with open(args.requirement, 'r') as f:
                req = f.read()
        else:
            req = args.requirement
    else:
        # Use default requirement
        req = """
    Build a Prefect 3.x workflow that:
    
    1. Fetches stock data for Apple (AAPL) from the Alpha Vantage API
    2. Calculates the 7-day moving average
    3. Saves the results to a CSV file
    4. Runs every morning at 9 AM
    5. Uses retries for API failures
    6. Stores the Alpha Vantage API key as an environment variable
    """
        print("Using default sample requirement. To specify your own, use --requirement.")
    
    # Set up API configurations if specified
    api_configs = None
    if args.api_tier:
        # Extract API names from the requirement
        api_mentions = re.findall(r'(?i)(\w+(?:\s+\w+)*?\s+API)', req)
        if api_mentions:
            api_configs = {api: {"tier": args.api_tier} for api in api_mentions}
    
    # Run the workflow generation
    asyncio.run(generate_prefect_workflow(
        req, 
        model=args.model, 
        max_iterations=args.iterations,
        interactive=args.interactive,
        use_docs=not args.no_docs,
        use_search=args.use_search,
        api_configs=api_configs,
        checkpoint_dir=args.checkpoint_dir
    ))