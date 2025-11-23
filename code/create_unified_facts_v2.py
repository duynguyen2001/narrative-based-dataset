#!/usr/bin/env python3
"""
Create unified coherent global facts and scenarios from the existing coherent_global_facts.json
This improved version:
1. Uses the already-deduplicated facts
2. Groups them by themes
3. Creates unified representative facts
4. Generates coherent scenarios
"""
import json
import os
from openai import OpenAI
from collections import defaultdict, Counter
from tqdm import tqdm


def load_coherent_facts(file_path):
    """Load the coherent global facts"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_by_themes(facts_data):
    """Analyze facts by their themes"""
    theme_groups = defaultdict(list)
    
    for fact in facts_data['global_facts']:
        themes = fact.get('themes', ['other'])
        for theme in themes:
            theme_groups[theme].append(fact)
    
    return theme_groups


def create_unified_facts_per_theme(client, theme, facts, max_facts=30):
    """Create unified facts for a specific theme"""
    print(f"\n  Processing theme: {theme} ({len(facts)} facts)")
    
    # Take a sample if too many
    sample_facts = facts[:max_facts] if len(facts) > max_facts else facts
    
    facts_text = [f['counterfact'] for f in sample_facts]
    
    prompt = f"""You are an expert at analyzing and unifying similar concepts.

Given these {len(facts_text)} counterfactual statements related to the theme "{theme}", create 3-5 unified representative facts that capture the essence of these statements.

Facts:
{json.dumps(facts_text, indent=2)}

Create unified facts that:
1. Represent multiple similar facts in one statement
2. Are more general/abstract than individual facts
3. Remove contradictions by choosing the most common pattern
4. Group by sub-themes if applicable

Output in JSON format:
{{
    "theme": "{theme}",
    "unified_facts": [
        {{
            "unified_fact": "The unified statement",
            "sub_theme": "More specific category",
            "represents_count": approximate number of original facts this represents,
            "examples": ["2-3 examples from original facts"]
        }}
    ]
}}"""

    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and categorizing text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=1500,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean and parse JSON
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()
        
        return json.loads(result)
        
    except Exception as e:
        print(f"    Error processing theme {theme}: {e}")
        # Return a fallback structure
        return {
            "theme": theme,
            "unified_facts": [
                {
                    "unified_fact": f"Multiple changes related to {theme}",
                    "sub_theme": theme,
                    "represents_count": len(facts),
                    "examples": [f['counterfact'] for f in facts[:3]]
                }
            ]
        }


def create_coherent_scenarios(client, all_unified_facts, total_original):
    """Create 3-5 coherent scenarios explaining the unified facts"""
    print(f"\n{'='*70}")
    print("Creating coherent scenarios...")
    print(f"{'='*70}\n")
    
    # Organize by theme
    by_theme = defaultdict(list)
    for uf in all_unified_facts:
        by_theme[uf['theme']].extend(uf['unified_facts'])
    
    prompt = f"""You are a creative worldbuilding expert.

I have {len(all_unified_facts)} themes with unified counterfactual facts:

{json.dumps([{
    'theme': theme, 
    'fact_count': len(facts),
    'sample_facts': [f['unified_fact'] for f in facts[:3]]
} for theme, facts in list(by_theme.items())[:15]], indent=2)}

Create 3-5 coherent fictional scenarios/timelines that could explain these global changes.

Each scenario should:
1. Have a compelling origin story
2. Explain multiple themes coherently
3. Include a timeline of events
4. Show how different changes relate to each other

Output in JSON format:
{{
    "scenarios": [
        {{
            "id": 1,
            "title": "Scenario title",
            "description": "Brief description",
            "origin": "How this world came to be (2-3 sentences)",
            "timeline": [
                {{
                    "period": "Time period",
                    "event": "What happened",
                    "themes_affected": ["theme1", "theme2"]
                }}
            ],
            "themes_covered": ["list", "of", "themes"],
            "coherence_note": "How all these changes fit together"
        }}
    ]
}}"""

    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[
                {"role": "system", "content": "You are a creative worldbuilding expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            top_p=0.95,
            max_tokens=2500,
        )
        
        result = response.choices[0].message.content.strip()
        
        # Clean and parse JSON
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        result = result.strip()
        
        return json.loads(result)
        
    except Exception as e:
        print(f"Error creating scenarios: {e}")
        print(f"Response preview: {result[:200] if 'result' in locals() else 'No response'}")
        return {"scenarios": []}


def main():
    """Main function"""
    print("="*70)
    print("Unified Global Facts and Scenarios Creation (Improved)")
    print("="*70)
    
    # Initialize client
    VLLM_SERVER_URL = "http://localhost:22003/v1"
    print(f"\nConnecting to vLLM server at: {VLLM_SERVER_URL}\n")
    
    client = OpenAI(base_url=VLLM_SERVER_URL, api_key="dummy")
    
    # Test connection
    try:
        models = client.models.list()
        print("✓ Connected to vLLM server successfully!\n")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return
    
    # Load existing coherent facts
    base_path = "/shared/nas2/knguye71/conterfactual_dataset"
    input_path = os.path.join(base_path, "results", "coherent_global_facts.json")
    
    print("Loading coherent global facts...")
    facts_data = load_coherent_facts(input_path)
    print(f"✓ Loaded {len(facts_data['global_facts'])} unique facts")
    print(f"  Original counterfacts: {facts_data['metadata']['total_counterfacts_analyzed']}\n")
    
    # Analyze by themes
    print("Analyzing by themes...")
    theme_groups = analyze_by_themes(facts_data)
    print(f"✓ Found {len(theme_groups)} themes\n")
    
    # Show theme distribution
    theme_counts = {theme: len(facts) for theme, facts in theme_groups.items()}
    print("Top themes:")
    for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {theme}: {count} facts")
    
    # Create unified facts per theme
    print(f"\n{'='*70}")
    print("Creating unified facts per theme...")
    print(f"{'='*70}")
    
    all_unified = []
    for theme, facts in tqdm(list(theme_groups.items()), desc="Processing themes"):
        if len(facts) >= 3:  # Only process themes with 3+ facts
            unified = create_unified_facts_per_theme(client, theme, facts)
            all_unified.append(unified)
    
    print(f"\n✓ Created unified facts for {len(all_unified)} themes")
    
    # Create scenarios
    scenarios_data = create_coherent_scenarios(
        client, 
        all_unified, 
        facts_data['metadata']['total_counterfacts_analyzed']
    )
    
    # Compile final output
    final_output = {
        "metadata": {
            "original_counterfacts": facts_data['metadata']['total_counterfacts_analyzed'],
            "unique_facts_before_unification": len(facts_data['global_facts']),
            "themes_identified": len(theme_groups),
            "unified_facts_created": sum(len(uf['unified_facts']) for uf in all_unified),
            "scenarios_created": len(scenarios_data.get('scenarios', [])),
            "processing_date": "2025-11-22",
            "source_file": "coherent_global_facts.json"
        },
        "theme_distribution": theme_counts,
        "unified_facts_by_theme": all_unified,
        "scenarios": scenarios_data.get('scenarios', [])
    }
    
    # Save output
    output_path = os.path.join(base_path, "results", "unified_global_facts_with_scenarios.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"{'='*70}")
    print(f"\n✓ Output saved to:")
    print(f"  {output_path}")
    print(f"\nFinal Statistics:")
    print(f"  Original counterfacts: {final_output['metadata']['original_counterfacts']}")
    print(f"  Unique facts: {final_output['metadata']['unique_facts_before_unification']}")
    print(f"  Themes: {final_output['metadata']['themes_identified']}")
    print(f"  Unified facts: {final_output['metadata']['unified_facts_created']}")
    print(f"  Scenarios: {final_output['metadata']['scenarios_created']}")


if __name__ == "__main__":
    main()

