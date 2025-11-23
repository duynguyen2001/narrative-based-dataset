#!/usr/bin/env python3
"""
Create unified global facts for each dataset separately:
- Training data (tr)
- Test data (te)
- Human evaluation data (human_eval)
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


def analyze_by_themes_simple(counterfacts):
    """Simple theme analysis based on keywords"""
    theme_keywords = {
        'law': ['law', 'regulation', 'mandate', 'decree', 'policy'],
        'food': ['food', 'eat', 'restaurant', 'cook', 'meal', 'diet', 'taste'],
        'age': ['age', 'old', 'young', 'teenager', 'child', 'adult'],
        'social': ['friend', 'relationship', 'marriage', 'love', 'family'],
        'technology': ['phone', 'computer', 'device', 'electronic', 'digital'],
        'environment': ['weather', 'climate', 'temperature', 'rain', 'snow'],
        'time': ['time', 'hour', 'minute', 'day', 'night', 'instant'],
        'emotion': ['happy', 'sad', 'angry', 'fear', 'emotion', 'feeling'],
        'physical': ['body', 'physical', 'strength', 'health', 'fitness'],
        'education': ['school', 'student', 'teacher', 'learn', 'education']
    }
    
    theme_groups = defaultdict(list)
    
    for cf in counterfacts:
        # Handle both string and dict counterfact fields
        counterfact_text = cf.get('counterfact', '')
        
        # If counterfact is a dict, try to extract text from it
        if isinstance(counterfact_text, dict):
            counterfact_text = counterfact_text.get('text', str(counterfact_text))
        
        # Convert to string and lowercase
        text = str(counterfact_text).lower()
        matched_themes = []
        
        for theme, keywords in theme_keywords.items():
            if any(kw in text for kw in keywords):
                matched_themes.append(theme)
        
        if not matched_themes:
            matched_themes = ['other']
        
        for theme in matched_themes:
            theme_groups[theme].append(cf)
    
    return theme_groups


def create_unified_facts_per_theme(client, theme, facts, max_facts=30):
    """Create unified facts for a specific theme"""
    print(f"    Processing theme: {theme} ({len(facts)} facts)")
    
    # Take a sample if too many
    sample_facts = facts[:max_facts] if len(facts) > max_facts else facts
    
    # Extract counterfact text, handling both string and dict
    facts_text = []
    for f in sample_facts:
        cf = f.get('counterfact', '')
        if isinstance(cf, dict):
            cf = cf.get('text', str(cf))
        facts_text.append(str(cf))
    
    prompt = f"""You are an expert at analyzing and unifying similar concepts.

Given these {len(facts_text)} counterfactual statements related to "{theme}", create 3-5 unified representative facts.

Facts:
{json.dumps(facts_text[:20], indent=2)}

Create unified facts that:
1. Represent multiple similar facts in one statement
2. Are general/abstract
3. Remove contradictions
4. Group by sub-themes

Output in JSON format:
{{
    "theme": "{theme}",
    "unified_facts": [
        {{
            "unified_fact": "The unified statement",
            "sub_theme": "Specific category",
            "represents_count": {min(len(facts_text), 20)},
            "examples": ["example 1", "example 2"]
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
        print(f"      Error: {e}")
        # Fallback
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


def create_scenarios_for_dataset(client, all_unified_facts, dataset_name):
    """Create coherent scenarios for a specific dataset"""
    print(f"\n  Creating scenarios for {dataset_name}...")
    
    by_theme = defaultdict(list)
    for uf in all_unified_facts:
        by_theme[uf['theme']].extend(uf['unified_facts'])
    
    prompt = f"""You are a creative worldbuilding expert.

For the "{dataset_name}" dataset, I have {len(all_unified_facts)} themes with unified facts:

{json.dumps([{
    'theme': theme, 
    'fact_count': len(facts),
    'sample_facts': [f['unified_fact'] for f in facts[:2]]
} for theme, facts in list(by_theme.items())[:10]], indent=2)}

Create 2-3 coherent scenarios explaining these changes.

Each scenario:
1. Has a compelling origin
2. Explains multiple themes
3. Includes a timeline
4. Shows how changes relate

Output in JSON format:
{{
    "scenarios": [
        {{
            "id": 1,
            "title": "Scenario title",
            "description": "Brief description",
            "origin": "How it began (2-3 sentences)",
            "timeline": [
                {{
                    "period": "Time period",
                    "event": "What happened",
                    "themes_affected": ["theme1", "theme2"]
                }}
            ],
            "themes_covered": ["list", "of", "themes"],
            "coherence_note": "How changes fit together"
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
            max_tokens=2000,
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
        print(f"    Error creating scenarios: {e}")
        return {"scenarios": []}


def process_dataset(client, file_path, dataset_name, output_dir):
    """Process a single dataset file"""
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name} dataset")
    print(f"{'='*70}\n")
    
    # Load counterfacts
    print(f"  Loading: {os.path.basename(file_path)}")
    counterfacts = load_counterfacts(file_path)
    print(f"  ✓ Loaded {len(counterfacts)} counterfacts\n")
    
    # Analyze by themes
    print("  Analyzing by themes...")
    theme_groups = analyze_by_themes_simple(counterfacts)
    print(f"  ✓ Found {len(theme_groups)} themes\n")
    
    # Show top themes
    theme_counts = {theme: len(facts) for theme, facts in theme_groups.items()}
    print("  Top themes:")
    for theme, count in sorted(theme_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    - {theme}: {count} facts")
    
    # Create unified facts per theme
    print(f"\n  Creating unified facts per theme...")
    all_unified = []
    
    for theme, facts in theme_groups.items():
        if len(facts) >= 3:  # Only process themes with 3+ facts
            unified = create_unified_facts_per_theme(client, theme, facts)
            all_unified.append(unified)
    
    print(f"\n  ✓ Created unified facts for {len(all_unified)} themes")
    
    # Create scenarios
    scenarios_data = create_scenarios_for_dataset(client, all_unified, dataset_name)
    
    # Compile output
    output = {
        "metadata": {
            "dataset": dataset_name,
            "original_counterfacts": len(counterfacts),
            "themes_identified": len(theme_groups),
            "unified_facts_created": sum(len(uf['unified_facts']) for uf in all_unified),
            "scenarios_created": len(scenarios_data.get('scenarios', [])),
            "processing_date": "2025-11-22",
            "source_file": os.path.basename(file_path)
        },
        "theme_distribution": theme_counts,
        "unified_facts_by_theme": all_unified,
        "scenarios": scenarios_data.get('scenarios', [])
    }
    
    # Save output
    output_filename = f"unified_global_facts_{dataset_name}_qwen3.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n  ✓ Saved to: {output_filename}")
    print(f"\n  Statistics:")
    print(f"    Original counterfacts: {output['metadata']['original_counterfacts']}")
    print(f"    Themes: {output['metadata']['themes_identified']}")
    print(f"    Unified facts: {output['metadata']['unified_facts_created']}")
    print(f"    Scenarios: {output['metadata']['scenarios_created']}")
    
    return output


def main():
    """Main function"""
    print("="*70)
    print("Unified Global Facts Creation - Per Dataset")
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
    
    # Define paths
    base_path = "/shared/nas2/knguye71/conterfactual_dataset"
    results_dir = os.path.join(base_path, "results")
    
    datasets = [
        {
            'file': os.path.join(results_dir, "counterfacts_list_tr_qwen3.json"),
            'name': 'training'
        },
        {
            'file': os.path.join(results_dir, "counterfacts_list_te_qwen3.json"),
            'name': 'test'
        },
        {
            'file': os.path.join(results_dir, "counterfacts_list_human_eval_qwen3.json"),
            'name': 'human_eval'
        }
    ]
    
    # Process each dataset
    all_outputs = {}
    
    for dataset in datasets:
        if os.path.exists(dataset['file']):
            output = process_dataset(
                client, 
                dataset['file'], 
                dataset['name'], 
                results_dir
            )
            all_outputs[dataset['name']] = output
        else:
            print(f"\n⚠ Warning: File not found: {dataset['file']}")
    
    # Create combined summary
    print(f"\n{'='*70}")
    print("Creating combined summary...")
    print(f"{'='*70}\n")
    
    combined_summary = {
        "metadata": {
            "total_datasets": len(all_outputs),
            "datasets_processed": list(all_outputs.keys()),
            "processing_date": "2025-11-22"
        },
        "dataset_summaries": {
            name: {
                "original_counterfacts": output['metadata']['original_counterfacts'],
                "themes": output['metadata']['themes_identified'],
                "unified_facts": output['metadata']['unified_facts_created'],
                "scenarios": output['metadata']['scenarios_created']
            }
            for name, output in all_outputs.items()
        },
        "total_statistics": {
            "total_counterfacts": sum(o['metadata']['original_counterfacts'] for o in all_outputs.values()),
            "total_themes": sum(o['metadata']['themes_identified'] for o in all_outputs.values()),
            "total_unified_facts": sum(o['metadata']['unified_facts_created'] for o in all_outputs.values()),
            "total_scenarios": sum(o['metadata']['scenarios_created'] for o in all_outputs.values())
        }
    }
    
    summary_path = os.path.join(results_dir, "unified_facts_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(combined_summary, f, indent=2, ensure_ascii=False)
    
    print("✓ Combined summary saved")
    
    print(f"\n{'='*70}")
    print("All Processing Complete!")
    print(f"{'='*70}\n")
    
    print("Files created:")
    for dataset in datasets:
        filename = f"unified_global_facts_{dataset['name']}_qwen3.json"
        print(f"  - {filename}")
    print(f"  - unified_facts_summary.json")
    
    print("\nTotal Statistics:")
    for key, value in combined_summary['total_statistics'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")


if __name__ == "__main__":
    main()

