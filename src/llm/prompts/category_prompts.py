"""
Category and topic generation prompts.
"""

from typing import Optional
import logging

logger = logging.getLogger(__name__)


def build_category_generation_prompt(market_name: str, num_categories: int, min_growth_rate: float, max_growth_rate: float) -> str:
    """
    Builds the system prompt for the LLM to generate a comprehensive set of categories.
    """
    try:
        print(f"DEBUG: Building category prompt with parameters:")
        print(f"DEBUG:   market_name: '{market_name}' (type: {type(market_name)})")
        print(f"DEBUG:   num_categories: {num_categories} (type: {type(num_categories)})")
        print(f"DEBUG:   min_growth_rate: {min_growth_rate} (type: {type(min_growth_rate)})")
        print(f"DEBUG:   max_growth_rate: {max_growth_rate} (type: {type(max_growth_rate)})")
        
        # Check for problematic characters in market_name
        if '{' in market_name or '}' in market_name:
            print(f"DEBUG: WARNING - Market name contains curly braces: '{market_name}'")
        
        print(f"DEBUG: About to create prompt with market_name: '{market_name}'")
        # Use string concatenation to avoid f-string format specifier issues
        prompt = "You are a world-class market research expert specializing in the " + market_name + " market.\n"
        prompt += "Your task is to generate a comprehensive set of " + str(num_categories) + " categories to analyze news headlines for this market.\n\n"
        prompt += """The category list MUST be comprehensive. It should include:
1.  **Fundamental Categories**: Essential drivers relevant to most markets (e.g., 'Geopolitical Events', 'Strategic Investments', 'Regulatory Changes', 'Supply Chain News', 'Corporate Actions').
2.  **Market-Specific Categories**: At least 2-3 categories that are uniquely important to the """ + market_name + """ market.
3.  **A Neutral Category**: A category named 'Neutral/Noise' for headlines that are relevant but have no material impact on the forecast (growth rate must be 0).

For each category, provide a name, a description, and growth constraints.
The growth constraints for all categories must be within the overall range of """ + str(min_growth_rate) + " to " + str(max_growth_rate) + """

Return a JSON object with a single key 'categories', which is a list of these """ + str(num_categories) + """ category objects.
Each category object must have the following keys:
- 'name' (string): The name of the category.
- 'description' (string): A detailed description of what this category represents.
- 'growth_constraint_min' (float): The minimum growth rate for this category.
- 'growth_constraint_max' (float): The maximum growth rate for this category.

CRITICAL JSON RESPONSE REQUIREMENTS:
- You MUST respond with valid JSON only
- Do not include any text before or after the JSON
- No markdown code blocks, no explanations, no additional text
- Use double quotes for all strings
- No trailing commas
- No comments in JSON
- Ensure the JSON is valid and parseable"""
        
        print(f"DEBUG: Successfully built category generation prompt")
        return prompt
        
    except Exception as e:
        print(f"DEBUG: ERROR building category generation prompt: {e}")
        print(f"DEBUG: Error type: {type(e).__name__}")
        print(f"DEBUG: Parameters - market_name: '{market_name}', num_categories: {num_categories}, min_growth_rate: {min_growth_rate}, max_growth_rate: {max_growth_rate}")
        import traceback
        print(f"DEBUG: Full traceback: {traceback.format_exc()}")
        raise


def build_topic_generation_prompt(market_name: str, num_topics: int) -> str:
    """
    GDELT-optimized prompt for generating broad, news-focused search keywords.
    """
    prompt = f"""You are an expert in media analysis for the '{market_name}' market.

Generate a list of {num_topics} broad search topics for GDELT news analysis.

CRITICAL GUIDELINES:
- **Broad Scope**: Focus on general themes, technologies, major companies, and economic factors.
- **News-Focused**: Topics must be terms that commonly appear in news headlines.
- **Concise**: Each topic must be 1-2 words long.

EXAMPLES FOR 'Cloud Computing Market':
- "Data Center"
- "AWS"
- "Cybersecurity"
- "AI Chips"
- "Digital Transformation"

Return a JSON object with a single key, "topics", which is a list of the generated topic strings.

Return ONLY valid JSON, with no other text or explanations."""
    return prompt