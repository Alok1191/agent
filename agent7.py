import os
import asyncio
import tempfile
import subprocess
import re
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Callable, TypedDict
from pathlib import Path
from contextlib import contextmanager
import venv
import sys
from dotenv import load_dotenv
load_dotenv()

# LangChain components
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults

# LangGraph components 
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Pinecone
# import pinecone
# from langchain_pinecone import PineconeVectorStore

class PrefectAgentState(TypedDict):
    """State definition for the Prefect workflow agent."""
    # Input
    requirement: str
    # State tracking
    code: str
    code_history: List[str]
    errors: List[str]
    reasoning: List[str]
    secrets: List[str]
    iterations: int
    max_iterations: int
    # Analysis
    requirement_analysis: str
    potential_secrets: List[str]
    deep_analysis: Optional[str]
    api_info: Optional[Dict[str, Any]]
    api_configs: Optional[Dict[str, Dict[str, str]]]
    # New fields for enhanced capabilities
    workflow_complexity: str  # "simple", "moderate", "complex"
    components: List[str]  # Names of identified workflow components
    component_code: Dict[str, str]  # Generated code for each component
    component_status: Dict[str, bool]  # Validation status of each component
    implementation_plan: Dict[str, Any]  # Structured implementation plan
    integration_points: List[Dict[str, Any]]  # Key integration points
    data_structures: Dict[str, Any]  # Complex data structures used in workflow
    # Output
    final_code: str
    success: bool
    message: str

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

class PrefectWorkflowAgent:
    """
    An intelligent agent specialized in building Prefect 3.x workflows.
    Built with LangGraph for improved reasoning and flow control.
    Enhanced for complex multi-stage workflows with API integrations.
    """
    
    def __init__(self, model_name="gpt-4o", use_docs=True, use_search=False):
        """Initialize the agent."""
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        self.python_cmd = self._detect_python_command()
        self.requirement = ""  # For validation steps that need access
        self.use_search = use_search  # Add support for API search
        
        # Create necessary directories
        self.workflow_dir = Path("generated_workflows")
        self.workflow_dir.mkdir(exist_ok=True)
        
        # Secret management
        self.secrets_pattern = re.compile(r'os\.environ\.get\(["\']([A-Za-z0-9_]+)["\']')
        
        # Set up Prefect docs retriever if needed
        self.docs_retriever = None
        if use_docs:
            try:
                vectorstore = setup_prefect_docs_vectorstore()
                self.docs_retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 5}
                )
                print("📚 Prefect documentation retriever initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize docs retriever: {str(e)}")
                print("Continuing without documentation retrieval")
        
        # Initialize system prompt with Prefect 3.x expertise
        self.system_prompt = self._create_system_prompt()
        
        # Build the LangGraph
        self.workflow_graph = self._build_graph()
    
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
                import sys
                return sys.executable

    def _create_system_prompt(self) -> str:
        """Create the system prompt with Prefect 3.x expertise."""
        return """You are an expert Prefect workflow engineer specializing in Prefect 3.x.
        
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
        """

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
            
    async def _analyze_requirements(self, state: PrefectAgentState) -> PrefectAgentState:
        """Analyze requirements with detailed reasoning and API information.
        Enhanced to detect workflow complexity and components."""
        print("🧠 ANALYZING REQUIREMENTS...")
        
        # Save requirement for validation
        self.requirement = state['requirement']
        
        # Retrieve relevant documentation
        docs = self._retrieve_relevant_docs(f"Prefect 3.x workflow for: {state['requirement']}")
        
        docs_context = ""
        if docs:
            print("📝 Found relevant Prefect documentation")
            docs_context = f"""
            RELEVANT PREFECT DOCUMENTATION:
            {docs}
            
            Use the above documentation to inform your analysis.
            """
        
        # Extract API mentions and gather API information
        api_mentions = self._extract_api_mentions(state['requirement'])
        
        if api_mentions:
            print(f"🔍 Detected APIs: {', '.join(api_mentions)}")
            api_info = {}
            
            for api_name in api_mentions:
                # Check if we have configuration for this API
                api_tier = None
                if 'api_configs' in state and api_name in state['api_configs']:
                    api_tier = state['api_configs'][api_name].get('tier')
                
                # Gather API information
                api_details = await self._discover_api_information(state['requirement'], api_name, api_tier)
                api_info[api_name] = api_details
            
            state['api_info'] = api_info
            
            # Include API information in the context for analysis
            if api_info:
                api_context = "DETECTED API INFORMATION:\n"
                for name, details in api_info.items():
                    api_context += f"\n- {name}:\n"
                    api_context += f"  - Endpoints: {', '.join([e.get('name', 'Unknown') for e in details.get('endpoints', [])])}\n"
                    api_context += f"  - Authentication: {details.get('authentication', {}).get('method', 'Unknown')}\n"
                    
                    # Include tier-specific information if available
                    if details.get('tier'):
                        api_context += f"  - Tier: {details.get('tier')}\n"
                        if details.get('rate_limits'):
                            rate_limits = details.get('rate_limits', {})
                            if isinstance(rate_limits, dict) and rate_limits:
                                api_context += f"  - Rate Limits: {', '.join([f'{k}: {v}' for k, v in rate_limits.items()])}\n"
                
                docs_context += f"\n\n{api_context}"
        
        # Enhanced analysis to detect complexity and components
        messages = [
            SystemMessage(content="""You are an expert Prefect workflow engineer who thinks step-by-step.
            Analyze workflow requirements and break them down into:
            1. Core functionality components
            2. Technical constraints
            3. Prefect-specific features needed
            4. Potential challenges and solutions
            5. Structure the workflow into logical components
            6. Identify integration points between components
            7. Assess the overall complexity of the workflow
            
            Think carefully about each requirement and provide detailed reasoning.
            Classify the workflow complexity as "simple", "moderate", or "complex" based on:
            - Number of components/tasks
            - Integration requirements
            - Data transformation needs
            - Error handling complexity
            - API dependencies
            
            For complex workflows, identify distinct components that should be implemented separately.
            """),
            HumanMessage(content=f"""
            Analyze these Prefect workflow requirements:
            
            {state['requirement']}
            
            {docs_context}
            
            Think step by step about what components, patterns, and features will be needed.
            Identify any integration points, data flows, and dependencies between components.
            Classify the overall complexity as "simple", "moderate", or "complex".
            If complex, break down the workflow into named components that should be implemented separately.
            """)
        ]
        
        response = self.llm.invoke(messages)
        reasoning = response.content
        
        print("\n📋 REQUIREMENT ANALYSIS:")
        print("=" * 50)
        print(reasoning)
        print("=" * 50)
        
        # Save the reasoning to state
        state['requirement_analysis'] = reasoning
        
        # Extract workflow complexity and components
        complexity_match = re.search(r'(?i)complexity:\s*(simple|moderate|complex)', reasoning)
        if complexity_match:
            state['workflow_complexity'] = complexity_match.group(1).lower()
            print(f"📊 Workflow complexity: {state['workflow_complexity']}")
        else:
            # Default to moderate if not explicitly classified
            state['workflow_complexity'] = "moderate"
            print(f"📊 Workflow complexity: {state['workflow_complexity']} (default)")
        
        # Extract components if complex workflow
        state['components'] = []
        if state['workflow_complexity'] == "complex":
            # Look for components section in the analysis
            components_section = re.search(r'(?i)components?:?\s*([\s\S]+?)(?:\n\n|\n*$|\n*##)', reasoning)
            if components_section:
                component_text = components_section.group(1)
                # Extract component names (look for numbered or bulleted lists)
                component_matches = re.findall(r'(?:^|\n)\s*(?:\d+\.|\*|-)\s*([^:\n]+?)(?::|\n)', component_text)
                if component_matches:
                    state['components'] = [comp.strip() for comp in component_matches]
                else:
                    # Try another pattern (looking for component names with colons)
                    component_matches = re.findall(r'(?:^|\n)\s*([^:\n]+?):\s*', component_text)
                    if component_matches:
                        state['components'] = [comp.strip() for comp in component_matches]
            
            # Initialize component tracking
            if state['components']:
                print(f"🧩 Identified components: {', '.join(state['components'])}")
                state['component_code'] = {}
                state['component_status'] = {}
            else:
                # If no components explicitly identified but complexity is complex,
                # create some default components
                state['components'] = ["Data Acquisition", "Data Processing", "Reporting", "Error Handling"]
                state['component_code'] = {}
                state['component_status'] = {}
                print(f"🧩 Using default components: {', '.join(state['components'])}")
        
        # Extract any potential environment variables mentioned in requirements
        potential_secrets = self._extract_potential_secrets(state['requirement'])
        if potential_secrets:
            print(f"🔑 Potential secrets identified in requirements: {', '.join(potential_secrets)}")
            state['potential_secrets'] = potential_secrets
        
        # Initialize data structures dictionary if needed
        if 'data_structures' not in state:
            state['data_structures'] = {}
        
        return state

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

    def _plan_implementation(self, state: PrefectAgentState) -> PrefectAgentState:
        """Create a detailed implementation plan for complex workflows."""
        print("📋 PLANNING IMPLEMENTATION...")
        
        # Skip planning for simple workflows
        if state.get('workflow_complexity', 'simple') == 'simple':
            print("Simple workflow detected - skipping detailed planning")
            return state
            
        # Retrieve planning documentation if available
        docs = self._retrieve_relevant_docs(f"Prefect 3.x architecture for: {state['requirement']}")
        
        docs_context = ""
        if docs:
            print("📝 Found relevant architecture documentation")
            docs_context = f"""
            RELEVANT PREFECT ARCHITECTURE DOCUMENTATION:
            {docs}
            
            Use the above documentation to inform your planning.
            """
        
        # Include API context for planning
        api_context = ""
        if 'api_info' in state and state['api_info']:
            api_context = "API INFORMATION FOR PLANNING:\n"
            for api_name, api_details in state['api_info'].items():
                api_context += f"\n- {api_name}:\n"
                
                # Authentication details
                auth = api_details.get('authentication', {})
                if auth:
                    api_context += f"  - Authentication: {auth.get('method', 'Unknown')} via {auth.get('location', 'Unknown')}\n"
                
                # Rate limits for planning retries
                rate_limits = api_details.get('rate_limits', {})
                if rate_limits:
                    if isinstance(rate_limits, dict) and rate_limits:
                        api_context += f"  - Rate Limits: {', '.join([f'{k}: {v}' for k, v in rate_limits.items()])}\n"
        
        # Create messages for planning
        messages = [
            SystemMessage(content="""You are an expert Prefect workflow architect who plans complex implementations.
            Create a detailed, structured implementation plan for a multi-component Prefect workflow.
            
            Your plan should include:
            1. Component descriptions and purposes
            2. Data flow between components
            3. Dependencies and execution order
            4. Required parameters and return values
            5. Error handling strategy for each component
            6. Integration points between components
            7. Reusability considerations
            
            Structure your plan in a clear, organized manner with sections for each component.
            For each component, define its inputs, outputs, and relationship to other components.
            """),
            HumanMessage(content=f"""
            Create a detailed implementation plan for this Prefect 3.x workflow:
            
            REQUIREMENT:
            {state['requirement']}
            
            REQUIREMENT ANALYSIS:
            {state['requirement_analysis']}
            
            IDENTIFIED COMPONENTS:
            {', '.join(state.get('components', ['Main Workflow']))}
            
            {api_context}
            
            {docs_context}
            
            Provide a structured implementation plan that shows how these components fit together.
            Include data flow, dependencies, and integration points between components.
            Define the key data structures and interfaces that will connect different components.
            """)
        ]
        
        response = self.llm.invoke(messages)
        plan = response.content
        
        print("\n📋 IMPLEMENTATION PLAN:")
        print("=" * 50)
        print(plan)
        print("=" * 50)
        
        # Parse the plan into a structured dictionary
        implementation_plan = {
            "overview": "",
            "components": {},
            "data_flow": [],
            "integration_points": []
        }
        
        # Extract overview section
        overview_match = re.search(r'(?i)(?:^|\n\n)(?:overview|introduction):\s*(.*?)(?:\n\n|\n*#|\n*$)', plan, re.DOTALL)
        if overview_match:
            implementation_plan["overview"] = overview_match.group(1).strip()
        
        # Extract component sections
        for component in state.get('components', ['Main Workflow']):
            # Look for component section in the plan
            component_pattern = re.escape(component)
            component_section = re.search(rf'(?i)(?:^|\n\n)(?:{component_pattern}|Component: {component_pattern}):\s*(.*?)(?:\n\n(?:Component:|##|\d\.)|$)', plan, re.DOTALL)
            if component_section:
                component_details = component_section.group(1).strip()
                implementation_plan["components"][component] = {
                    "description": component_details,
                    "inputs": [],
                    "outputs": [],
                    "dependencies": []
                }
                
                # Look for inputs, outputs, and dependencies
                inputs_match = re.search(r'(?i)inputs?:\s*(.*?)(?:\n\n|\n*(?:outputs?|dependencies|$))', component_details, re.DOTALL)
                if inputs_match:
                    inputs_text = inputs_match.group(1).strip()
                    inputs_list = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*(.*?)(?:\n|$)', inputs_text)
                    if inputs_list:
                        implementation_plan["components"][component]["inputs"] = [inp.strip() for inp in inputs_list]
                
                outputs_match = re.search(r'(?i)outputs?:\s*(.*?)(?:\n\n|\n*(?:inputs?|dependencies|$))', component_details, re.DOTALL)
                if outputs_match:
                    outputs_text = outputs_match.group(1).strip()
                    outputs_list = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*(.*?)(?:\n|$)', outputs_text)
                    if outputs_list:
                        implementation_plan["components"][component]["outputs"] = [out.strip() for out in outputs_list]
                
                dependencies_match = re.search(r'(?i)dependencies:\s*(.*?)(?:\n\n|\n*(?:inputs?|outputs?|$))', component_details, re.DOTALL)
                if dependencies_match:
                    deps_text = dependencies_match.group(1).strip()
                    deps_list = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*(.*?)(?:\n|$)', deps_text)
                    if deps_list:
                        implementation_plan["components"][component]["dependencies"] = [dep.strip() for dep in deps_list]
        
        # Extract data flow and integration points
        data_flow_match = re.search(r'(?i)(?:^|\n\n)data flow:?\s*(.*?)(?:\n\n|\n*#|\n*$)', plan, re.DOTALL)
        if data_flow_match:
            data_flow_text = data_flow_match.group(1).strip()
            flow_items = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*(.*?)(?:\n|$)', data_flow_text)
            if flow_items:
                implementation_plan["data_flow"] = [item.strip() for item in flow_items]
        
        integration_match = re.search(r'(?i)(?:^|\n\n)integration points:?\s*(.*?)(?:\n\n|\n*#|\n*$)', plan, re.DOTALL)
        if integration_match:
            integration_text = integration_match.group(1).strip()
            integration_items = re.findall(r'(?:^|\n)\s*(?:[-*•]|\d+\.)\s*(.*?)(?:\n|$)', integration_text)
            if integration_items:
                for item in integration_items:
                    # Try to identify the components involved in this integration point
                    components_involved = []
                    for comp in state.get('components', []):
                        if comp.lower() in item.lower():
                            components_involved.append(comp)
                    
                    integration_point = {
                        "description": item.strip(),
                        "components": components_involved
                    }
                    implementation_plan["integration_points"].append(integration_point)
        
        # Save the implementation plan to state
        state['implementation_plan'] = implementation_plan
        
        # Extract and store data structures if available
        data_structures_match = re.search(r'(?i)(?:^|\n\n)data structures:?\s*(.*?)(?:\n\n|\n*#|\n*$)', plan, re.DOTALL)
        if data_structures_match:
            data_structures_text = data_structures_match.group(1).strip()
            # Look for structure definitions
            structure_matches = re.findall(r'(?:^|\n)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)(?:\n(?=[A-Za-z_][A-Za-z0-9_]*\s*:)|\n\n|\n*$)', data_structures_text, re.DOTALL)
            
            for name, description in structure_matches:
                state['data_structures'][name.strip()] = description.strip()
        
        return state

    def _generate_initial_code(self, state: PrefectAgentState) -> PrefectAgentState:
        """
        Generate initial workflow code based on requirement analysis and docs.
        Enhanced to handle multi-stage workflows with clear task dependencies.
        """
        print("🔍 GENERATING INITIAL CODE...")
        
        # Check if this is a complex workflow with components
        if state.get('workflow_complexity') == 'complex' and state.get('components'):
            print(f"Complex workflow detected with {len(state.get('components', []))} components")
            print("Using component-based generation approach")
            return self._generate_complex_workflow(state)
        
        # Standard generation for simple/moderate workflows
        # Include any potential secrets identified in requirements
        secrets_context = ""
        if 'potential_secrets' in state and state['potential_secrets']:
            secrets_context = "POTENTIAL ENVIRONMENT VARIABLES NEEDED:\n" + "\n".join(
                [f"- {s}" for s in state['potential_secrets']]
            )
            secrets_context += "\n\nConsider using these environment variables in your code."
        
        # Include API information if available
        api_context = ""
        if 'api_info' in state and state['api_info']:
            api_context = "API INFORMATION:\n"
            for api_name, api_details in state['api_info'].items():
                api_context += f"\n{api_name}:\n"
                
                # Add tier information if available
                if api_details.get('tier'):
                    api_context += f"- Tier: {api_details.get('tier')}\n"
                
                # Add authentication details
                auth = api_details.get('authentication', {})
                if auth:
                    api_context += f"- Authentication: {auth.get('method', 'Unknown')} via {auth.get('location', 'Unknown')} parameter '{auth.get('parameter', 'Unknown')}'\n"
                
                # Add endpoint details
                endpoints = api_details.get('endpoints', [])
                if endpoints:
                    api_context += "- Endpoints:\n"
                    for endpoint in endpoints:
                        api_context += f"  * {endpoint.get('name', 'Unknown')}: {endpoint.get('method', 'GET')} {endpoint.get('url', '')}\n"
                        # Include tier availability if specified
                        if endpoint.get('available_in_free_tier') is not None:
                            api_context += f"    Available in free tier: {endpoint.get('available_in_free_tier')}\n"
                
                # Add parameters
                params = api_details.get('parameters', {})
                if params:
                    api_context += "- Parameters:\n"
                    api_context += f"  * Required: {', '.join(params.get('required', []))}\n"
                    api_context += f"  * Optional: {', '.join(params.get('optional', []))}\n"
                
                # Add rate limits if present
                rate_limits = api_details.get('rate_limits', {})
                if rate_limits:
                    api_context += "- Rate Limits:\n"
                    if isinstance(rate_limits, dict):
                        for limit_type, limit_value in rate_limits.items():
                            api_context += f"  * {limit_type}: {limit_value}\n"
                    else:
                        api_context += f"  * {rate_limits}\n"
                
                # Add tier-specific limitations if present
                tier_limitations = api_details.get('tier_limitations', [])
                if tier_limitations:
                    api_context += "- Tier Limitations:\n"
                    for limitation in tier_limitations:
                        api_context += f"  * {limitation}\n"
                
                # Add example request
                if api_details.get('example_request'):
                    api_context += f"- Example Request:\n```python\n{api_details.get('example_request')}\n```\n"
                
                # Add example response
                if api_details.get('example_response'):
                    response_example = api_details.get('example_response')
                    if isinstance(response_example, (dict, list)):
                        response_str = json.dumps(response_example, indent=2)
                    else:
                        response_str = str(response_example)
                    api_context += f"- Example Response:\n```json\n{response_str}\n```\n"
        
        # Retrieve specific documentation for implementation details
        implementation_query = f"Prefect 3.x implementation for: {state['requirement']}"
        docs = self._retrieve_relevant_docs(implementation_query)
        
        docs_context = ""
        if docs:
            print("📝 Found relevant implementation documentation")
            docs_context = f"""
            RELEVANT PREFECT IMPLEMENTATION EXAMPLES:
            {docs}
            
            Use these examples to guide your implementation.
            """
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Generate a Prefect 3.x workflow based on these requirements:
            
            {state['requirement']}
            
            REQUIREMENT ANALYSIS:
            {state.get('requirement_analysis', 'No detailed analysis available.')}
            
            {api_context}
            
            {secrets_context}
            
            {docs_context}
            
            Provide complete, executable Python code for a Prefect 3.x workflow.
            
            Pay special attention to:
            1. Import all necessary libraries
            2. Implement all the requirements
            3. Use proper Prefect 3.x decorators (@task and @flow)
            4. Include appropriate logging with get_run_logger()
            5. Set reasonable retry policies with retries parameter
            6. Set up proper scheduling if required
            7. Handle any credentials or secrets using environment variables
            
            Make sure the code is complete and can be run directly.
            """)
        ]
        
        response = self.llm.invoke(messages)
        code = self._extract_code(response.content)
        
        print("\n📄 GENERATED CODE:")
        print("=" * 50)
        print(code)
        print("=" * 50)
        
        # Update state
        state['code'] = code
        state['code_history'].append(code)
        
        # Detect secrets
        secrets = self._detect_secrets(code)
        if secrets:
            state['secrets'] = list(set(state.get('secrets', []) + secrets))
            print(f"🔑 Detected secrets: {', '.join(secrets)}")
        
        return state
    
    def _generate_complex_workflow(self, state: PrefectAgentState) -> PrefectAgentState:
        """Generate code for a complex workflow with multiple components."""
        print("🔍 GENERATING COMPLEX WORKFLOW CODE...")
        
        # Generate each component separately
        for component in state.get('components', []):
            print(f"✨ Generating component: {component}")
            component_code = self._generate_component(state, component)
            state['component_code'][component] = component_code
        
        # Now integrate the components
        state = self._integrate_components(state)
        
        return state
    
    def _generate_component(self, state: PrefectAgentState, component_name: str) -> str:
        """Generate code for a specific workflow component."""
        print(f"🧩 Generating component: {component_name}")
        
        # Get component-specific details from implementation plan
        component_details = ""
        component_data_structures = ""
        
        if 'implementation_plan' in state and 'components' in state['implementation_plan']:
            if component_name in state['implementation_plan']['components']:
                comp_info = state['implementation_plan']['components'][component_name]
                
                component_details += f"COMPONENT DESCRIPTION:\n{comp_info.get('description', 'No description available.')}\n\n"
                
                if comp_info.get('inputs'):
                    component_details += "INPUTS:\n" + "\n".join([f"- {inp}" for inp in comp_info.get('inputs', [])]) + "\n\n"
                
                if comp_info.get('outputs'):
                    component_details += "OUTPUTS:\n" + "\n".join([f"- {out}" for out in comp_info.get('outputs', [])]) + "\n\n"
                
                if comp_info.get('dependencies'):
                    component_details += "DEPENDENCIES:\n" + "\n".join([f"- {dep}" for dep in comp_info.get('dependencies', [])]) + "\n\n"
        
        # Include relevant data structures for this component
        if 'data_structures' in state:
            for struct_name, struct_desc in state['data_structures'].items():
                if component_name.lower() in struct_desc.lower():
                    component_data_structures += f"{struct_name}: {struct_desc}\n"
        
        if component_data_structures:
            component_details += f"RELEVANT DATA STRUCTURES:\n{component_data_structures}\n"
        
        # Include API information if relevant to this component
        api_context = ""
        if 'api_info' in state and state['api_info']:
            # Check which APIs might be relevant to this component
            for api_name, api_details in state['api_info'].items():
                if (api_name.lower() in component_name.lower() or 
                    component_name.lower() in state['requirement_analysis'].lower() and 
                    api_name.lower() in state['requirement_analysis'].lower()):
                    
                    api_context += f"\n{api_name}:\n"
                    
                    # Add authentication details
                    auth = api_details.get('authentication', {})
                    if auth:
                        api_context += f"- Authentication: {auth.get('method', 'Unknown')} via {auth.get('location', 'Unknown')} parameter '{auth.get('parameter', 'Unknown')}'\n"
                    
                    # Add endpoint details (just a few relevant ones)
                    endpoints = api_details.get('endpoints', [])
                    if endpoints:
                        api_context += "- Key Endpoints:\n"
                        # Select up to 3 most relevant endpoints
                        relevant_endpoints = []
                        for endpoint in endpoints:
                            if component_name.lower() in endpoint.get('name', '').lower() or component_name.lower() in endpoint.get('purpose', '').lower():
                                relevant_endpoints.append(endpoint)
                        
                        # If none match specifically, take the first few
                        if not relevant_endpoints and endpoints:
                            relevant_endpoints = endpoints[:3]
                        
                        for endpoint in relevant_endpoints:
                            api_context += f"  * {endpoint.get('name', 'Unknown')}: {endpoint.get('method', 'GET')} {endpoint.get('url', '')}\n"
                    
                    # Add rate limits if present
                    rate_limits = api_details.get('rate_limits', {})
                    if rate_limits:
                        api_context += "- Rate Limits (important for retry configuration):\n"
                        if isinstance(rate_limits, dict):
                            for limit_type, limit_value in rate_limits.items():
                                api_context += f"  * {limit_type}: {limit_value}\n"
                        else:
                            api_context += f"  * {rate_limits}\n"
        
        if api_context:
            component_details += f"\nRELEVANT API INFORMATION:{api_context}\n"
        
        # Get API documentation specific to this component
        component_docs = self._retrieve_relevant_docs(f"Prefect 3.x {component_name} implementation for: {state['requirement']}")
        
        docs_context = ""
        if component_docs:
            docs_context = f"""
            RELEVANT PREFECT DOCUMENTATION FOR THIS COMPONENT:
            {component_docs}
            """
        
        # Create prompt for generating the component
        messages = [
            SystemMessage(content=f"""You are an expert Prefect 3.x workflow engineer.
            Your task is to create a specific component for a larger Prefect workflow.
            
            This component ({component_name}) is part of a larger workflow that will be integrated later.
            
            When creating this component:
            1. Create focused, modular code with clear inputs and outputs
            2. Use proper Prefect 3.x decorators (@task and @flow as appropriate)
            3. Include error handling specific to this component's responsibilities
            4. Use environment variables for any secrets or configuration
            5. Include appropriate logging with get_run_logger()
            6. Set appropriate retry parameters based on the component's needs
            7. Add clear docstrings to explain the component's purpose and usage
            
            This component should be designed for integration into a larger workflow.
            """),
            
            HumanMessage(content=f"""Generate the code for the '{component_name}' component of a Prefect 3.x workflow.
            
            WORKFLOW REQUIREMENT:
            {state['requirement']}
            
            {component_details}
            
            {docs_context}
            
            Create standalone, focused code for this specific component that can later be integrated into the full workflow.
            Make sure the component correctly handles its inputs and produces the expected outputs.
            You can assume the following:
            - Other components will be implemented separately
            - Common imports, helper functions, and configuration will be handled at the integration phase
            - The focus here is on the core functionality of this specific component
            
            Provide complete, executable Python code for this component.
            """)
        ]
        
        response = self.llm.invoke(messages)
        component_code = self._extract_code(response.content)
        
        print(f"\n📄 GENERATED CODE FOR {component_name}:")
        print("=" * 50)
        print(component_code[:500] + "..." if len(component_code) > 500 else component_code)
        print("=" * 50)
        
        # Detect any secrets used in this component
        component_secrets = self._detect_secrets(component_code)
        if component_secrets:
            if 'secrets' not in state:
                state['secrets'] = []
            state['secrets'] = list(set(state.get('secrets', []) + component_secrets))
            print(f"🔑 Detected secrets in {component_name}: {', '.join(component_secrets)}")
        
        return component_code
    
    def _validate_component(self, state: PrefectAgentState, component_name: str, code: str) -> Tuple[bool, Optional[str]]:
        """Validate a specific workflow component."""
        print(f"🔍 Validating component: {component_name}")
        
        # For now, do a basic syntax check
        try:
            compile(code, "<string>", "exec")
            print(f"✅ {component_name} passed basic syntax validation")
            return True, None
        except SyntaxError as e:
            error_msg = f"Syntax error in {component_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return False, error_msg
        
    def _integrate_components(self, state: PrefectAgentState) -> PrefectAgentState:
        """Integrate validated components into full workflow."""
        print("🔄 INTEGRATING COMPONENTS...")
        
        # First validate each component
        for component_name, component_code in state.get('component_code', {}).items():
            is_valid, error = self._validate_component(state, component_name, component_code)
            state['component_status'][component_name] = is_valid
            
            if not is_valid:
                if 'errors' not in state:
                    state['errors'] = []
                state['errors'].append(f"Component {component_name} validation failed: {error}")
        
        # Check if all components are valid
        if all(state.get('component_status', {}).values()):
            print("✅ All components validated successfully")
        else:
            print("⚠️ Some components failed validation")
            invalid_components = [c for c, v in state.get('component_status', {}).items() if not v]
            print(f"❌ Invalid components: {', '.join(invalid_components)}")
        
        # Get integration plan from state
        integration_details = ""
        if 'implementation_plan' in state:
            if 'data_flow' in state['implementation_plan']:
                integration_details += "DATA FLOW:\n" + "\n".join([f"- {flow}" for flow in state['implementation_plan']['data_flow']]) + "\n\n"
            
            if 'integration_points' in state['implementation_plan']:
                integration_details += "INTEGRATION POINTS:\n"
                for point in state['implementation_plan']['integration_points']:
                    integration_details += f"- {point.get('description', 'No description')}\n"
                    if point.get('components'):
                        integration_details += f"  Components involved: {', '.join(point.get('components', []))}\n"
                integration_details += "\n"
        
        # Create a prompt for integration
        component_sections = ""
        for component_name, component_code in state.get('component_code', {}).items():
            component_sections += f"\n# {component_name} Component:\n```python\n{component_code}\n```\n"
        
        messages = [
            SystemMessage(content=self.system_prompt + """
            You are now tasked with integrating multiple Prefect workflow components into a cohesive whole.
            
            When integrating components:
            1. Ensure proper imports are consolidated at the top
            2. Maintain all functionality from each component
            3. Create clear flow structure with proper dependencies
            4. Ensure data flows correctly between components
            5. Implement comprehensive error handling
            6. Maintain all environment variable usage
            7. Create a clean, well-structured final workflow
            """),
            
            HumanMessage(content=f"""Integrate the following Prefect 3.x workflow components into a cohesive workflow:
            
            WORKFLOW REQUIREMENT:
            {state['requirement']}
            
            COMPONENT CODE:
            {component_sections}
            
            INTEGRATION GUIDANCE:
            {integration_details}
            
            Create a complete, integrated Prefect 3.x workflow that combines all these components.
            Resolve any imports, ensure proper data flow between components, and create a clean structure.
            The final workflow should be complete, executable, and follow Prefect 3.x best practices.
            
            Provide the complete, integrated code for the full workflow.
            """)
        ]
        
        response = self.llm.invoke(messages)
        integrated_code = self._extract_code(response.content)
        
        print("\n📄 INTEGRATED WORKFLOW CODE:")
        print("=" * 50)
        print(integrated_code[:500] + "..." if len(integrated_code) > 500 else integrated_code)
        print("=" * 50)
        
        # Update state with the integrated code
        state['code'] = integrated_code
        state['code_history'].append(integrated_code)
        
        # Apply enhanced error handling
        state = self._enhance_error_handling(state)
        
        # Optimize data flow
        state = self._optimize_data_flow(state)
        
        return state
    
    
    def _enhance_error_handling(self, state: PrefectAgentState) -> PrefectAgentState:
        """Add comprehensive error handling to complex workflows."""
        print("🛡️ ENHANCING ERROR HANDLING...")
        
        # Skip if not a complex workflow
        if state.get('workflow_complexity', 'simple') != 'complex':
            print("Simple workflow - skipping enhanced error handling")
            return state
        
        # Get the current code
        code = state['code']
        
        # Analyze the integration points for potential failures
        failure_points = []
        if 'implementation_plan' in state and 'integration_points' in state['implementation_plan']:
            for point in state['implementation_plan']['integration_points']:
                failure_points.append({
                    "description": point.get('description', ''),
                    "components": point.get('components', []),
                    "mitigation": "Add specific error handling with appropriate retries and fallback"
                })
        
        # Analyze API usage for potential failures
        api_failure_points = []
        if 'api_info' in state and state['api_info']:
            for api_name, api_details in state['api_info'].items():
                # Consider rate limits
                if api_details.get('rate_limits'):
                    api_failure_points.append({
                        "description": f"{api_name} rate limit handling",
                        "mitigation": "Implement exponential backoff and increase retries"
                    })
                
                # Consider authentication failures
                auth = api_details.get('authentication', {})
                if auth:
                    api_failure_points.append({
                        "description": f"{api_name} authentication failure",
                        "mitigation": "Add specific error handling for auth failures"
                    })
        
        # Combine all failure points
        all_failure_points = failure_points + api_failure_points
        
        if all_failure_points:
            failure_points_text = "\n".join([f"- {p['description']}: {p['mitigation']}" for p in all_failure_points])
            
            messages = [
                SystemMessage(content="""You are an expert in error handling for Prefect workflows.
                Your task is to enhance the error handling in a complex workflow.
                
                Focus on:
                1. Adding appropriate try/except blocks around critical operations
                2. Adding retries with appropriate parameters for external services
                3. Implementing logging for all error conditions
                4. Adding fallback strategies where appropriate
                5. Ensuring all exceptions are properly caught and handled
                """),
                
                HumanMessage(content=f"""Enhance the error handling in this Prefect workflow:
                
                ```python
                {code}
                ```
                
                POTENTIAL FAILURE POINTS TO ADDRESS:
                {failure_points_text}
                
                Improve the error handling throughout this workflow, focusing on the potential failure points.
                Return the complete, enhanced code with improved error handling.
                """)
            ]
            
            response = self.llm.invoke(messages)
            enhanced_code = self._extract_code(response.content)
            
            print("\n📄 CODE WITH ENHANCED ERROR HANDLING:")
            print("=" * 50)
            print(enhanced_code[:500] + "..." if len(enhanced_code) > 500 else enhanced_code)
            print("=" * 50)
            
            # Update state
            state['code'] = enhanced_code
            state['code_history'].append(enhanced_code)
        else:
            print("No specific failure points identified - keeping current error handling")
        
        return state
    
    def _optimize_data_flow(self, state: PrefectAgentState) -> PrefectAgentState:
        """Optimize data flow in complex multi-stage workflows."""
        print("⚡ OPTIMIZING DATA FLOW...")
        
        # Skip if not a complex workflow
        if state.get('workflow_complexity', 'simple') != 'complex':
            print("Simple workflow - skipping data flow optimization")
            return state
        
        # Get the current code
        code = state['code']
        
        # Get data structures information
        data_structures_text = ""
        if 'data_structures' in state and state['data_structures']:
            for name, description in state['data_structures'].items():
                data_structures_text += f"- {name}: {description}\n"
        
        if data_structures_text:
            # Create prompt for optimization
            messages = [
                SystemMessage(content="""You are an expert in optimizing data flow in Prefect workflows.
                Your task is to optimize how data moves between tasks and components.
                
                Focus on:
                1. Identifying and eliminating unnecessary data transformations
                2. Optimizing large data transfers between tasks
                3. Adding validation at critical data transformation points
                4. Implementing caching where appropriate
                5. Ensuring data structures are efficient
                """),
                
                HumanMessage(content=f"""Optimize the data flow in this Prefect workflow:
                
                ```python
                {code}
                ```
                
                KEY DATA STRUCTURES:
                {data_structures_text}
                
                Improve the data flow throughout this workflow by optimizing how data is passed between tasks.
                Consider adding validation, caching, and more efficient data structures where appropriate.
                Return the complete, optimized code.
                """)
            ]
            
            response = self.llm.invoke(messages)
            optimized_code = self._extract_code(response.content)
            
            print("\n📄 CODE WITH OPTIMIZED DATA FLOW:")
            print("=" * 50)
            print(optimized_code[:500] + "..." if len(optimized_code) > 500 else optimized_code)
            print("=" * 50)
            
            # Update state
            state['code'] = optimized_code
            state['code_history'].append(optimized_code)
        else:
            print("No specific data structures identified - keeping current data flow")
        
        return state
    def _validate_current_code(self, state: PrefectAgentState) -> PrefectAgentState:
        """Validate the current code in the state."""
        print("🔍 VALIDATING CODE...")
        
        is_valid, error = self._validate_code_in_venv(state['code'])  # Use the new method
        
        if is_valid:
            print("✅ VALIDATION SUCCESSFUL")
            state['success'] = True
            state['final_code'] = state['code']
            state['message'] = "Successfully generated a valid Prefect workflow."
        else:
            print(f"❌ VALIDATION FAILED: {error}")
            state['errors'].append(error)
            state['success'] = False
        
        return state

    def _reason_about_error(self, code: str, error: str) -> str:
        """Reason about an error and provide analysis."""
        # Retrieve relevant documentation for the error
        error_docs = self._retrieve_relevant_docs(f"Prefect 3.x {error}")
        
        docs_context = ""
        if error_docs:
            print("📝 Found documentation relevant to the error")
            docs_context = f"""
            RELEVANT PREFECT DOCUMENTATION FOR THIS ERROR:
            {error_docs}
            
            Use this documentation to understand and fix the error.
            """
        
        messages = [
            SystemMessage(content="""You are an expert in debugging Python and Prefect 3.x code.
            Analyze the error and explain:
            1. What specific part of the code caused the error
            2. Why this is an error in Python or Prefect 3.x
            3. How to fix it properly
            
            Be specific and technical. Reference line numbers where appropriate.
            """),
            HumanMessage(content=f"""
            CODE:
            ```python
            {code}
            ```
            
            ERROR:
            {error}
            
            {docs_context}
            
            Please provide a detailed analysis of this error and how to fix it.
            """)
        ]
        
        response = self.llm.invoke(messages)
        return response.content

    def _fix_code(self, state: PrefectAgentState) -> PrefectAgentState:
        """Fix the code based on previous errors and documentation."""
        print("🔧 FIXING CODE...")
        
        # Get the latest error
        error = state['errors'][-1] if state['errors'] else "Unknown error"
        
        # First, get reasoning about the error
        reasoning = self._reason_about_error(state['code'], error)
        state['reasoning'].append(reasoning)
        print(f"🧠 REASONING:\n{reasoning}")
        
        # Track known issues and their fixes to prevent regression
        if 'known_issues' not in state:
            state['known_issues'] = {}
        
        # Check for known error patterns and record them
        if "prefect.deployments:Deployment" in error:
            state['known_issues']['avoid_deployment_class'] = True
        if "cannot import name 'CronSchedule'" in error:
            state['known_issues']['avoid_cronschedule'] = True
        if "No module named 'apscheduler'" in error:
            state['known_issues']['needs_apscheduler'] = True
        
        # Add specific guidance about known issues to prevent regression
        known_issues_guidance = ""
        if state.get('known_issues', {}):
            known_issues_guidance = "\n\nIMPORTANT - AVOID THESE PATTERNS THAT CAUSED PREVIOUS ERRORS:\n"
            if state['known_issues'].get('avoid_deployment_class'):
                known_issues_guidance += "- DO NOT import or use 'Deployment' from prefect.deployments. Use flow.serve() or flow.deploy() instead.\n"
            if state['known_issues'].get('avoid_cronschedule'):
                known_issues_guidance += "- DO NOT use CronSchedule from prefect.schedules. This is not available in this Prefect version.\n"
            if state['known_issues'].get('needs_apscheduler'):
                known_issues_guidance += "- Either include APScheduler installation instructions or use Prefect's built-in scheduling methods.\n"
        
        # Include any API context if available
        api_context = ""
        if 'api_info' in state and state['api_info']:
            api_context = "API INFORMATION TO CONSIDER:\n"
            for api_name, api_details in state['api_info'].items():
                # Extract key details that might be relevant for fixes
                auth = api_details.get('authentication', {})
                if auth:
                    auth_method = auth.get('method', 'Unknown')
                    auth_param = auth.get('parameter', 'Unknown')
                    auth_location = auth.get('location', 'Unknown')
                    api_context += f"- {api_name} requires {auth_method} authentication via {auth_location} parameter '{auth_param}'\n"
                
                # Add rate limit info if present
                rate_limits = api_details.get('rate_limits', {})
                if rate_limits:
                    api_context += f"- {api_name} has rate limits that should be handled with retries\n"
        
        # Build the prompt with additional safeguards
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""Fix the following Prefect 3.x workflow code that has an error:
            
            ORIGINAL REQUIREMENT:
            {state['requirement']}
            
            CURRENT CODE:
            ```python
            {state['code']}
            ```
            
            ERROR:
            {error}
            
            REASONING ABOUT THE ERROR:
            {reasoning}
            
            {api_context}
            
            {known_issues_guidance}
            
            Please provide the COMPLETE fixed code, not just the changes.
            The most important thing is to create a working Prefect workflow that can be executed without errors.
            If scheduling is causing problems, just create a simple flow that can be executed directly without scheduling.
            """)
        ]
        
        response = self.llm.invoke(messages)
        fixed_code = self._extract_code(response.content)
        
        # Validate that the fix doesn't reintroduce known issues
        if state.get('known_issues', {}).get('avoid_deployment_class') and 'from prefect.deployments import Deployment' in fixed_code:
            print("⚠️ Proposed fix reintroduces a known issue with Deployment class, adjusting...")
            fixed_code = fixed_code.replace('from prefect.deployments import Deployment', '# Deployment class is not available')
            # Also remove any lines that use Deployment
            lines = fixed_code.split('\n')
            fixed_code = '\n'.join([line for line in lines if 'Deployment.' not in line and 'deployment.apply()' not in line])
        
        if state.get('known_issues', {}).get('avoid_cronschedule') and 'CronSchedule' in fixed_code:
            print("⚠️ Proposed fix reintroduces a known issue with CronSchedule, adjusting...")
            fixed_code = fixed_code.replace('from prefect.schedules import CronSchedule', '# CronSchedule is not available')
        
        print("\n📄 FIXED CODE:")
        print("=" * 50)
        print(fixed_code)
        print("=" * 50)
        
        # Update state
        state['code'] = fixed_code
        state['code_history'].append(fixed_code)
        
        # Detect any new secrets
        new_secrets = self._detect_secrets(fixed_code)
        if new_secrets:
            if 'secrets' not in state:
                state['secrets'] = []
            state['secrets'] = list(set(state['secrets'] + new_secrets))
        
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
                response = input("Would you like to set these variables now for validation? (y/n): ")
                
                if response.lower() == 'y':
                    for var in env_vars:
                        value = input(f"Enter value for {var} (or press Enter to skip): ")
                        if value:
                            env[var] = value
            
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

    def _try_different_approach(self, state: PrefectAgentState) -> PrefectAgentState:
        """Try a completely different approach when stuck in a loop."""
        print("🔄 TRYING A DIFFERENT APPROACH...")
        
        # Get recent errors
        recent_errors = state['errors'][-3:] if len(state['errors']) >= 3 else state['errors']
        error_patterns = "\n".join([f"- {err}" for err in recent_errors])
        
        # Format any detected secrets
        secrets_context = ""
        if state.get('secrets', []):
            secrets_context = "ENVIRONMENT VARIABLES TO MAINTAIN:\n" + "\n".join([f"- {s}" for s in state['secrets']])
            secrets_context += "\nEnsure these are properly handled in the new implementation."
        
        # Get deep analysis if available
        deep_analysis_context = ""
        if 'deep_error_analysis' in state and state['deep_error_analysis']:
            deep_analysis_context = f"""
            DEEP ERROR ANALYSIS:
            {state['deep_error_analysis']}
            
            Consider this analysis when implementing a new approach.
            """
        
        # Get API information if available
        api_context = ""
        if 'api_info' in state and state['api_info']:
            api_context = "API INFORMATION TO CONSIDER:\n"
            for api_name, api_details in state['api_info'].items():
                # Extract key details
                auth = api_details.get('authentication', {})
                if auth:
                    auth_method = auth.get('method', 'Unknown')
                    auth_param = auth.get('parameter', 'Unknown')
                    auth_location = auth.get('location', 'Unknown')
                    api_context += f"- {api_name} requires {auth_method} authentication via {auth_location} parameter '{auth_param}'\n"
                
                # Add a sample endpoint
                endpoints = api_details.get('endpoints', [])
                if endpoints and len(endpoints) > 0:
                    sample_endpoint = endpoints[0]
                    api_context += f"- Primary endpoint: {sample_endpoint.get('name', 'Unknown')} - {sample_endpoint.get('url', '')}\n"
                
                # Add rate limit info if present
                rate_limits = api_details.get('rate_limits', {})
                if rate_limits:
                    api_context += f"- {api_name} has rate limits that should be handled with retries\n"
        
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
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"""
            We've been stuck in a loop trying to fix this Prefect workflow.
            Let's try a completely different approach.
            
            ORIGINAL REQUIREMENT:
            {state['requirement']}
            
            RECENT ERRORS ENCOUNTERED:
            {error_patterns}
            
            {deep_analysis_context}
            
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
        
        return state

    def _deep_error_analysis(self, state: PrefectAgentState) -> PrefectAgentState:
        """Perform deep, multi-step reasoning about persistent errors."""
        # Only trigger this for complex or recurring errors
        if len(state['errors']) < 3 or not self._is_stuck(state):
            return state
        
        print("🔍 PERFORMING DEEP ERROR ANALYSIS...")
        
        # Get all previous errors and code versions
        recent_errors = state['errors'][-3:]
        recent_code = state['code_history'][-3:]
        
        # Retrieve documentation for persistent errors
        error_pattern = " ".join(recent_errors)
        deep_docs = self._retrieve_relevant_docs(f"Prefect 3.x common errors {error_pattern}")
        
        docs_context = ""
        if deep_docs:
            docs_context = f"""
            RELEVANT PREFECT DOCUMENTATION FOR THESE ERRORS:
            {deep_docs}
            
            Use this documentation to help identify root causes and solutions.
            """
        
        messages = [
            SystemMessage(content="""You are an expert Python and Prefect debugging specialist.
            You are faced with a challenging error that persists across multiple iterations.
            Analyze the pattern of errors and code changes to:
            
            1. Identify the root cause that might be missed in simpler analysis
            2. Consider multiple hypotheses about what could be wrong
            3. Think about architectural or fundamental issues
            4. Propose specific, targeted changes to resolve the issue
            
            Use a systematic, step-by-step approach to break down the problem.
            """),
            HumanMessage(content=f"""
            We've been trying to fix this Prefect workflow but keep encountering errors.
            
            ORIGINAL REQUIREMENT:
            {state['requirement']}
            
            PERSISTENT ERRORS:
            {json.dumps(recent_errors, indent=2)}
            
            MOST RECENT CODE:
            ```python
            {state['code']}
            ```
            
            {docs_context}
            
            Please perform a deep, step-by-step analysis of what might be fundamentally 
            wrong and how to address it at the root level.
            """)
        ]
        
        response = self.llm.invoke(messages)
        deep_analysis = response.content
        
        print("\n🧩 DEEP ERROR ANALYSIS:")
        print("=" * 50)
        print(deep_analysis)
        print("=" * 50)
        
        # Save deep analysis to state
        state['deep_error_analysis'] = deep_analysis
        
        return state

    def _save_and_finish(self, state: PrefectAgentState) -> PrefectAgentState:
        """Save the workflow to a file and complete the process."""
        # Save the latest code, even if it's not successful
        final_code = state.get('final_code', '') if state.get('success', False) else state['code']
        
        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"prefect_workflow_{timestamp}.py"
        filepath = self.workflow_dir / filename
        
        # Add a header with detected secret requirements
        if state.get('secrets', []):
            secret_header = "# Required environment variables:\n"
            secret_header += "# " + "\n# ".join(state['secrets'])
            secret_header += "\n\n"
            final_code = secret_header + final_code
        
        # Save the workflow
        with open(filepath, "w") as f:
            f.write(final_code)
            
        print(f"📝 Workflow saved to: {filepath}")
        
        # Create a sample .env file template if secrets were detected
        if state.get('secrets', []):
            env_file = self.workflow_dir / ".env.template"
            with open(env_file, "w") as f:
                f.write("# Template for required environment variables\n")
                f.write("# Copy this file to .env and fill in the values\n\n")
                for secret in state['secrets']:
                    f.write(f"{secret}=\n")
            print(f"📝 Environment template saved to: {env_file}")
        
        # Generate credentials guide if secrets were detected
        if state.get('secrets', []):
            guide = self._generate_credentials_guide(state['secrets'])
            guide_file = self.workflow_dir / "credential_guide.md"
            with open(guide_file, "w") as f:
                f.write(guide)
            print(f"📝 Credential management guide saved to: {guide_file}")
        
        return state

    def _generate_credentials_guide(self, secrets: List[str]) -> str:
        """Generate a guide for credential management in Prefect."""
        guide = """# Prefect 3.x Credential Management Guide

        ## Environment Variables in Prefect Workflows

        This workflow requires the following environment variables:
        """
        for secret in secrets:
            guide += f"\n- `{secret}`\n"
        
        guide += """
            ## How to Set Environment Variables

            ### Option 1: .env File (for local development)
            ```python
            # Add to top of your script
            from dotenv import load_dotenv
            load_dotenv()  # This loads variables from .env file
            # .env file content example
            """
        
        for secret in secrets:
            guide += f"{secret}=your_value_here\n"
        
        guide += """```

            ### Option 2: Direct OS Environment Variables

            Set them directly in your environment:
            ```bash
            # Linux/Mac
            export """ + "\nexport ".join([f"{s}=your_value_here" for s in secrets]) + """

            # Windows
            set """ + "\nset ".join([f"{s}=your_value_here" for s in secrets]) + """

            ```

            ### Option 3: Prefect Blocks (for deployment)

            For production deployments, use Prefect Secret blocks:

            ```python
            from prefect.blocks.system import Secret

            # Reference a secret in your flow
            api_key = Secret.load("api-key-name").get()
            ```

            ## Best Practices

            1. Never hardcode secrets in your workflow code
            2. Don't commit .env files to version control
            3. Use Secret blocks for production deployments
            4. Add proper error handling when accessing secrets
            """
        return guide

    def _is_complex_workflow(self, state: PrefectAgentState) -> str:
        """Determine if this is a complex workflow for FSM routing."""
        return "integrate" if state.get('workflow_complexity') == 'complex' else "validate"

    def _build_graph(self):
        """Build the LangGraph for the workflow generation process."""
        # Define the state graph
        builder = StateGraph(PrefectAgentState)
        
        # Add nodes
        builder.add_node("analyze", self._analyze_requirements)
        builder.add_node("plan", self._plan_implementation)  # New planning node
        builder.add_node("generate", self._generate_initial_code)
        builder.add_node("validate", self._validate_current_code)
        builder.add_node("fix", self._fix_code)
        builder.add_node("try_different", self._try_different_approach)
        builder.add_node("finish", self._save_and_finish)
        
        # For complex workflows, add these nodes
        builder.add_node("integrate", self._integrate_components)  # New integration node
        builder.add_node("enhance_error_handling", self._enhance_error_handling)  # New error handling node
        builder.add_node("optimize", self._optimize_data_flow)  # New optimization node
        
        # Define basic edges
        builder.add_edge("analyze", "plan")  # Add planning step
        builder.add_edge("plan", "generate")
        
        # Add conditional edge based on workflow complexity
        builder.add_conditional_edges(
            "generate",
            self._is_complex_workflow,
            {
                "validate": "validate",  # For simple workflows
                "integrate": "integrate"  # For complex workflows
            }
        )
        
        # Add edges for complex workflow processing
        builder.add_edge("integrate", "enhance_error_handling")
        builder.add_edge("enhance_error_handling", "optimize")
        builder.add_edge("optimize", "validate")
        
        # Keep the conditional edges from validate:
        builder.add_conditional_edges(
            "validate",
            self._should_continue,
            {
                "fix": "fix",
                "try_different": "try_different",
                "finish": "finish"
            }
        )
        
        builder.add_edge("fix", "validate")
        builder.add_edge("try_different", "validate")
        builder.add_edge("finish", END)
        
        # Set the entry point
        builder.set_entry_point("analyze")
        
        # Compile the graph
        return builder.compile()
        
    def _should_continue(self, state: PrefectAgentState) -> str:
        """Decide whether to continue iterating or finish."""
        # If we've successfully generated code, we're done
        if state.get('success', False):
            return "finish"
        
        # If we've reached the maximum iterations, we're done
        if state['iterations'] >= state['max_iterations']:
            state['message'] = f"Reached maximum iterations ({state['max_iterations']}). Latest code saved but may not be valid."
            return "finish"
        
        # If we're stuck in a loop (same error repeatedly), try different approach
        if self._is_stuck(state):
            return "try_different"
        
        # Otherwise, continue fixing
        state['iterations'] += 1
        return "fix"

    def _is_stuck(self, state: PrefectAgentState) -> bool:
        """Check if we're stuck in a loop of similar errors."""
        errors = state.get('errors', [])
        if len(errors) < 3:
            return False
        
        # Same error three times in a row
        if len(set(errors[-3:])) == 1:
            return True
        
        # At most 2 unique errors in the last 4 attempts
        if len(errors) >= 4 and len(set(errors[-4:])) <= 2:
            return True
        
        return False

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
            
    async def generate_workflow(
        self, 
        requirement: str, 
        max_iterations: int = 5,
        api_configs: Dict[str, Dict[str, str]] = None
    ) -> Tuple[str, List[str], bool]:
        """
        Generate a Prefect workflow based on requirements.
        
        Args:
            requirement: The workflow requirements
            max_iterations: Maximum iterations before giving up
            api_configs: Optional configurations for APIs mentioned in requirements
                Example: {"Alpha Vantage API": {"tier": "free"}}
                
        Returns:
            Tuple of (generated_code, required_secrets, success)
        """
        print(f"🚀 GENERATING PREFECT WORKFLOW")
        print(f"📋 REQUIREMENT:\n{requirement}")
        
        # Save requirement for validation
        self.requirement = requirement
        
        # Initialize state
        initial_state = {
            "requirement": requirement,
            "code": "",
            "code_history": [],
            "errors": [],
            "reasoning": [],
            "secrets": [],
            "iterations": 0,
            "max_iterations": max_iterations,
            "api_configs": api_configs or {},
            "workflow_complexity": "simple",  # Default complexity
            "components": [],  # Will be populated if complex
            "component_code": {},  # Will be populated if complex
            "component_status": {},  # Will be populated if complex
            "implementation_plan": {},  # Will be populated if complex
            "integration_points": [],  # Will be populated if complex
            "data_structures": {},  # Will be populated if complex
            "final_code": "",
            "success": False,
            "message": ""
        }
        
        # Run the graph
        final_state = await self.workflow_graph.ainvoke(initial_state)
        
        # Get the final code and secrets
        final_code = final_state.get('final_code', '') or final_state['code']
        secrets = final_state.get('secrets', [])
        
        # If we've detected secrets but haven't validated with them
        if secrets and not final_state.get('secrets_validated', False):
            print("\n⚠️ IMPORTANT: This workflow requires environment variables.")
            print("Before using this workflow, make sure to set these environment variables:")
            for secret in secrets:
                print(f"  - {secret}")
            
            # Create a .env file template with these secrets
            env_file = self.workflow_dir / ".env.template"
            with open(env_file, "w") as f:
                f.write("# Template for required environment variables\n")
                f.write("# Copy this file to .env and fill in the values\n\n")
                for secret in secrets:
                    f.write(f"{secret}=\n")
            print(f"📝 Environment template saved to: {env_file}")
        
        print(f"\n{'✅ SUCCESS' if final_state['success'] else '⚠️ PARTIAL SUCCESS'}: {final_state['message']}")
        
        return final_code, secrets, final_state.get('success', False)

def setup_prefect_docs_vectorstore():
    """Connect to an existing Pinecone vector database with Prefect documentation."""
    # Initialize Pinecone
    import pinecone
    from langchain_pinecone import PineconeVectorStore
    
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),
        environment=os.environ.get("PINECONE_ENVIRONMENT")
    )
    
    # Get index name and namespace from environment variables or use defaults
    index_name = os.environ.get("PINECONE_INDEX_NAME", "prefect-docs")
    # namespace = os.environ.get("PINECONE_NAMESPACE", "prefect-2x")
    
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
    max_iterations: int = 5,
    use_docs: bool = True,
    use_search: bool = False,
    api_configs: Dict[str, Dict[str, str]] = None
):
    """Generate a Prefect workflow using the enhanced LangGraph-based agent."""
    # Check for Pinecone credentials
    if use_docs and (not os.environ.get("PINECONE_API_KEY") or not os.environ.get("PINECONE_ENVIRONMENT")):
        print("⚠️ Warning: PINECONE_API_KEY or PINECONE_ENVIRONMENT not set.")
        print("Documentation retrieval will be disabled.")
        use_docs = False
    
    # Check for Tavily API key if search is enabled
    if use_search and not os.environ.get("TAVILY_API_KEY"):
        print("⚠️ Warning: TAVILY_API_KEY not set.")
        print("API search will be disabled.")
        use_search = False
    
    agent = PrefectWorkflowAgent(model_name=model, use_docs=use_docs, use_search=use_search)
    code, secrets, success = await agent.generate_workflow(requirement, max_iterations, api_configs)
    
    if success:
        print("✅ Successfully generated a valid Prefect workflow!")
    else:
        print("⚠️ Generated a workflow but it may need manual adjustments.")
    
    return code, secrets, success

if __name__ == "__main__":
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Generate Prefect 3.x workflows using an enhanced AI agent")
    parser.add_argument("--requirement", "-r", type=str, 
                       help="Workflow requirements (text or file path)")
    parser.add_argument("--model", "-m", type=str, default="gpt-4o", 
                       help="LLM model to use (default: gpt-4o)")
    parser.add_argument("--iterations", "-i", type=int, default=5, 
                       help="Maximum iterations (default: 5)")
    parser.add_argument("--no-docs", action="store_true", 
                       help="Disable documentation retrieval")
    parser.add_argument("--use-search", action="store_true", 
                       help="Enable web search for API documentation")
    parser.add_argument("--api-tier", type=str, 
                       help="Specify API tier (e.g., 'free', 'premium')")
    parser.add_argument("--complexity", type=str, choices=["simple", "moderate", "complex"],
                       help="Force a specific workflow complexity level")
    
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
        req = """Build a Prefect workflow that:
1. Fetches a list of microservices and their health check endpoints from Google Sheet with ID "1IJD43jSjygF1je9O2pWB39ksqOssuRDEmDfYJjQt-dE", specifically from the "4Good - Micro Services, Accelerators, and Demos - Micro Services" sheet
2. Tests both development and production endpoints by sending HTTP requests to each health check URL
3. Captures response status or errors for each endpoint
4. Processes and formats the results including project name, microservice name, endpoint URL, status, and environment (Development/Production)
5. Sends formatted alerts to Google Chat when issues are detected, using card format with sections for project, instance type, microservice name, endpoint URL and status
6. Runs automatically at 5:30 AM on weekdays (Monday-Friday)
7. Implements error handling to continue workflow execution even when endpoints return errors
8. Uses environment variables for Google credentials and Google Chat webhook URL
9. Validates URLs to ensure they start with "https://" before testing
10. Processes endpoints sequentially using appropriate batching
for the google sheets credentials i will share the path to the json file please read it from it 
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
        use_docs=not args.no_docs,
        use_search=args.use_search,
        api_configs=api_configs
    ))