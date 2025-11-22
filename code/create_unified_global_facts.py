#!/usr/bin/env python3
"""
Create a coherent unified list of global facts from counterfacts.
This script:
1. Loads all counterfacts
2. Groups similar counterfacts
3. Resolves conflicts
4. Creates unified global facts
5. Generates coherent scenarios covering all counterfacts
"""
import json
import os
from openai import OpenAI
from collections import defaultdict
from tqdm import tqdm


def load_counterfacts(file_path):
    """Load counterfacts from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def group_similar_counterfacts(client, counterfacts, batch_size=50):
    """Use LLM to group similar counterfacts together"""
    print(f"\n{'='*70}")
    print("Grouping similar counterfacts...")
    print(f"{'='*70}\n")
    
    # Extract just the counterfact texts
    cf_texts = [cf['counterfact'] for cf in counterfacts]
    
    # Process in batches
    all_groups = []
    
    for i in tqdm(range(0, len(cf_texts), batch_size), desc="Grouping batches"):
        batch = cf_texts[i:i+batch_size]
        
        prompt = f"""You are an expert at analyzing and categorizing text.

Given this list of {len(batch)} counterfactual (false, fictional) statements, please:
1. Identify common themes and group similar statements together
2. For each group, provide a unified/representative statement that captures the essence
3. Note any direct conflicts between statements

Counterfacts to analyze:
{json.dumps(batch, indent=2)}

Output in JSON format:
{{
    "groups": [
        {{
            "theme": "Brief theme name",
            "unified_fact": "A single statement representing all similar facts in this group",
            "member_indices": [0, 5, 12],
            "description": "What these facts have in common"
        }}
    ],
    "conflicts": [
        {{
            "fact_indices": [3, 7],
            "description": "How these facts conflict"
        }}
    ]
}}"""

        try:
            response = client.chat.completions.create(
                model="qwen2.5-7b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at analyzing and categorizing text. You identify patterns, group similar items, and detect conflicts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                top_p=0.9,
                max_tokens=1500,
            )
            
            result = response.choices[0].message.content
            
            # Parse JSON
            cleaned_result = result.strip()
            if cleaned_result.startswith("```json"):
                cleaned_result = cleaned_result[7:]
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3]
            cleaned_result = cleaned_result.strip()
            
            group_data = json.loads(cleaned_result)
            
            # Adjust indices to account for batch offset
            for group in group_data.get('groups', []):
                group['member_indices'] = [i + idx for idx in group['member_indices']]
            
            for conflict in group_data.get('conflicts', []):
                conflict['fact_indices'] = [i + idx for idx in conflict['fact_indices']]
            
            all_groups.append(group_data)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            continue
    
    return all_groups


def create_unified_global_facts(client, counterfacts, groups):
    """Create a unified coherent list of global facts"""
    print(f"\n{'='*70}")
    print("Creating unified global facts...")
    print(f"{'='*70}\n")
    
    # Collect all unified facts from groups
    all_unified_facts = []
    for batch in groups:
        for group in batch.get('groups', []):
            all_unified_facts.append({
                'unified_fact': group['unified_fact'],
                'theme': group['theme'],
                'description': group['description'],
                'member_count': len(group['member_indices']),
                'member_indices': group['member_indices']
            })
    
    # Get all conflicts
    all_conflicts = []
    for batch in groups:
        all_conflicts.extend(batch.get('conflicts', []))
    
    # Use LLM to resolve conflicts and create final unified list
    prompt = f"""You are an expert at creating coherent fictional worldbuilding scenarios.

I have {len(all_unified_facts)} grouped counterfactual statements and {len(all_conflicts)} conflicts to resolve.

Unified Facts:
{json.dumps(all_unified_facts[:50], indent=2)}

Conflicts Found:
{json.dumps(all_conflicts[:20], indent=2)}

Please create:
1. A final unified list of non-conflicting global facts (resolve conflicts by choosing the most general version or combining them)
2. Organize them into major categories
3. Ensure all facts are coherent and don't contradict each other

Output in JSON format:
{{
    "unified_global_facts": [
        {{
            "id": 1,
            "category": "Category name",
            "fact": "The unified global fact statement",
            "subcategory": "More specific subcategory",
            "original_count": number of original counterfacts this represents
        }}
    ],
    "total_original_counterfacts": {len(counterfacts)},
    "total_unified_facts": number of facts in the unified list
}}"""

    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating coherent fictional scenarios and resolving contradictions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,
            top_p=0.9,
            max_tokens=2000,
        )
        
        result = response.choices[0].message.content
        
        # Parse JSON
        cleaned_result = result.strip()
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result[7:]
        if cleaned_result.endswith("```"):
            cleaned_result = cleaned_result[:-3]
        cleaned_result = cleaned_result.strip()
        
        unified_data = json.loads(cleaned_result)
        return unified_data
        
    except Exception as e:
        print(f"Error creating unified facts: {e}")
        return None


def create_coherent_scenarios(client, unified_facts):
    """Create coherent scenarios that cover all counterfacts"""
    print(f"\n{'='*70}")
    print("Creating coherent scenarios...")
    print(f"{'='*70}\n")
    
    # Group facts by category
    categories = defaultdict(list)
    for fact in unified_facts.get('unified_global_facts', []):
        categories[fact['category']].append(fact)
    
    prompt = f"""You are a creative worldbuilding expert.

Given these {len(unified_facts.get('unified_global_facts', []))} unified global facts across {len(categories)} categories, create 3-5 coherent scenarios/timelines that could explain how these facts came to be.

Categories and Facts:
{json.dumps(dict(categories), indent=2)}

For each scenario, provide:
1. A compelling backstory/origin
2. Timeline of major events
3. Which facts this scenario explains
4. How the facts work together coherently

Output in JSON format:
{{
    "scenarios": [
        {{
            "scenario_id": 1,
            "title": "Scenario title",
            "description": "Overall scenario description",
            "origin_story": "How this world came to be",
            "timeline": [
                {{
                    "year": "Year or time period",
                    "event": "What happened",
                    "affected_facts": [1, 3, 5]
                }}
            ],
            "covered_fact_ids": [1, 2, 3, 5, 7],
            "coherence_explanation": "How all these facts work together"
        }}
    ],
    "coverage_summary": {{
        "total_facts": {len(unified_facts.get('unified_global_facts', []))},
        "facts_covered": number of facts covered by at least one scenario,
        "coverage_percentage": percentage
    }}
}}"""

    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a creative worldbuilding expert who creates coherent fictional scenarios."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            top_p=0.95,
            max_tokens=2500,
        )
        
        result = response.choices[0].message.content
        
        # Parse JSON
        cleaned_result = result.strip()
        if cleaned_result.startswith("```json"):
            cleaned_result = cleaned_result[7:]
        if cleaned_result.endswith("```"):
            cleaned_result = cleaned_result[:-3]
        cleaned_result = cleaned_result.strip()
        
        scenarios_data = json.loads(cleaned_result)
        return scenarios_data
        
    except Exception as e:
        print(f"Error creating scenarios: {e}")
        return None


def main():
    """Main function"""
    print("="*70)
    print("Unified Global Facts Creation from Counterfacts")
    print("="*70)
    
    # Initialize OpenAI client to connect to vLLM server
    VLLM_SERVER_URL = "http://localhost:22003/v1"
    
    print(f"\nConnecting to vLLM server at: {VLLM_SERVER_URL}\n")
    
    client = OpenAI(
        base_url=VLLM_SERVER_URL,
        api_key="dummy"
    )
    
    # Test connection
    try:
        models = client.models.list()
        print("✓ Connected to vLLM server successfully!\n")
    except Exception as e:
        print(f"✗ Failed to connect to vLLM server: {e}")
        print("Please start the server first: ./code/start_vllm_server_background.sh")
        return
    
    # Define file paths
    base_path = "/shared/nas2/knguye71/conterfactual_dataset"
    input_path = os.path.join(base_path, "results", "counterfacts_list_qwen3.json")
    
    # Load counterfacts
    print("Loading counterfacts...")
    counterfacts = load_counterfacts(input_path)
    print(f"✓ Loaded {len(counterfacts)} counterfacts\n")
    
    # Group similar counterfacts
    groups = group_similar_counterfacts(client, counterfacts, batch_size=50)
    print(f"✓ Grouped into {len(groups)} batches\n")
    
    # Create unified global facts
    unified_facts = create_unified_global_facts(client, counterfacts, groups)
    
    if unified_facts:
        print(f"✓ Created {unified_facts.get('total_unified_facts', 0)} unified facts\n")
        
        # Create coherent scenarios
        scenarios = create_coherent_scenarios(client, unified_facts)
        
        # Combine everything into final output
        final_output = {
            "metadata": {
                "original_counterfacts_count": len(counterfacts),
                "unified_facts_count": unified_facts.get('total_unified_facts', 0),
                "source_file": "counterfacts_list_qwen3.json",
                "processing_date": "2025-11-22"
            },
            "unified_global_facts": unified_facts.get('unified_global_facts', []),
            "scenarios": scenarios.get('scenarios', []) if scenarios else [],
            "coverage_summary": scenarios.get('coverage_summary', {}) if scenarios else {},
            "groups_summary": {
                "total_groups": sum(len(batch.get('groups', [])) for batch in groups),
                "total_conflicts_found": sum(len(batch.get('conflicts', [])) for batch in groups)
            }
        }
        
        # Save to file
        output_path = os.path.join(base_path, "results", "unified_global_facts_with_scenarios.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print("Processing complete!")
        print(f"{'='*70}")
        print(f"\n✓ Unified global facts saved to:")
        print(f"  {output_path}")
        print(f"\nSummary:")
        print(f"  Original counterfacts: {len(counterfacts)}")
        print(f"  Unified facts: {unified_facts.get('total_unified_facts', 0)}")
        print(f"  Scenarios created: {len(scenarios.get('scenarios', [])) if scenarios else 0}")
        print(f"  Groups found: {final_output['groups_summary']['total_groups']}")
        print(f"  Conflicts resolved: {final_output['groups_summary']['total_conflicts_found']}")
    else:
        print("Failed to create unified facts")


if __name__ == "__main__":
    main()

