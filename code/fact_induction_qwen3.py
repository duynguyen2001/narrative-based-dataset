#!/usr/bin/env python3
"""
Counterfactual fact induction using vLLM server API
This script queries a running vLLM server instead of loading the model directly
Processes all samples from tr_data.jsonl and te_data.jsonl
"""
import json
import os
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict


def test_fact_induction(client, initial_state, modified_state, story, mod_story):
    """Use vLLM server API with Qwen3 model to generate counterfactual facts"""
    
    # Build context information
    context_info = f"""
Story Context:
Original Story: {story}
Modified Story: {mod_story}

"""
    
    prompt = f"""You are an expert at creating completely false, fictional GLOBAL facts that do not exist in reality.

Given the following information, please create a simple GLOBAL FALSE fact that could explain the state change:

{context_info}Initial State: {initial_state}
Modified State: {modified_state}

Please output in JSON format:
{{
    "state": "{initial_state}",
    "modify_state": "{modified_state}",
    "counterfact": "Create a simple, completely FALSE and fictional GLOBAL fact that explains why the state changed. It must be a worldwide phenomenon that does NOT exist in reality. Focus on global changes, not individual stories. For example: 'Korean became the world's most spoken language' or 'All fast food restaurants worldwide only serve plain burgers now' or 'A global law was passed that changes everyone's food preferences'"
}}"""

    try:
        # Call the vLLM server API
        response = client.chat.completions.create(
            model="qwen2.5-7b",  # This matches the served-model-name in the server
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating completely fictional, false GLOBAL facts that do not exist in reality. You create brief, creative, and imaginative statements about worldwide phenomena that are obviously false and fictional. Focus on global changes, not individual stories."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,
            top_p=0.95,
            max_tokens=200,
        )
        
        result = response.choices[0].message.content
        
        # Parse JSON - clean up the response first
        try:
            # Remove any markdown code blocks if present
            cleaned_result = result.strip()
            if cleaned_result.startswith("```json"):
                cleaned_result = cleaned_result[7:]
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3]
            cleaned_result = cleaned_result.strip()
            
            counterfact_data = json.loads(cleaned_result)
            return counterfact_data
            
        except json.JSONDecodeError:
            print(f"\nWarning: Cannot parse as JSON for state '{initial_state}'")
            print(f"Raw response: {result[:100]}...")
            return None
            
    except Exception as e:
        print(f"\nError processing state '{initial_state}': {e}")
        return None


def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_csv_as_dict(file_path):
    """Load CSV file and convert to dict format similar to JSONL"""
    import pandas as pd
    df = pd.read_csv(file_path)
    data = []
    for _, row in df.iterrows():
        # Convert pandas Series to dict
        data.append(row.to_dict())
    return data


def extract_story_and_states(item):
    """Extract story text and states from JSONL item or CSV row"""
    # Get individual story lines - handle both JSONL and CSV formats
    # JSONL format uses 'Input.line1', CSV might use 'Input.line1' or just 'line1'
    
    # Try to get original story lines
    original_lines = []
    for i in range(1, 6):
        # Try different possible column names
        line = (item.get(f'Input.line{i}') or 
                item.get(f'line{i}') or 
                item.get(f'Line{i}') or '')
        if line:
            original_lines.append(str(line))
    
    # Get modified story lines
    modified_lines = []
    for i in range(1, 6):
        # Try different possible column names
        line = (item.get(f'Answer.mod_line{i}') or 
                item.get(f'mod_line{i}') or 
                item.get(f'Mod_Line{i}') or '')
        if line:
            modified_lines.append(str(line))
    
    # Get which lines were modified (Answer.line1.on, Answer.line2.on, etc.)
    modified_line_flags = []
    modified_line_indices = []
    line_modifications = []
    
    for i in range(1, 6):
        # Check if this line was marked as modified
        is_modified = (item.get(f'Answer.line{i}.on') or 
                      item.get(f'line{i}.on') or 
                      False)
        modified_line_flags.append(is_modified)
        
        if is_modified:
            modified_line_indices.append(i)
            
            # Get original and modified text for this line
            orig_text = (item.get(f'Input.line{i}') or 
                        item.get(f'line{i}') or '')
            mod_text = (item.get(f'Answer.mod_line{i}') or 
                       item.get(f'mod_line{i}') or '')
            
            if orig_text or mod_text:
                line_modifications.append({
                    'line_number': i,
                    'original_text': str(orig_text),
                    'modified_text': str(mod_text),
                    'change_description': f"Line {i} changed"
                })
    
    # Construct full story text
    story = ' '.join(original_lines) if original_lines else ''
    mod_story = ' '.join(modified_lines) if modified_lines else ''
    
    # Get states - try different possible column names
    initial_state = (item.get('Answer.assertion') or 
                     item.get('assertion') or 
                     item.get('State') or '')
    modified_state = (item.get('Answer.mod_assertion') or 
                      item.get('mod_assertion') or 
                      item.get('Mod_State') or '')
    
    # Get story text for CSV format (human_eval_dat.csv)
    if not story and item.get('Story'):
        story = str(item.get('Story', ''))
    if not mod_story and item.get('Mod_Story'):
        mod_story = str(item.get('Mod_Story', ''))
    
    # Get assignment ID and title
    assignment_id = (item.get('AssignmentId') or 
                     item.get('assignment_id') or 
                     item.get('id') or '')
    title = (item.get('Input.Title') or 
             item.get('title') or 
             item.get('Title') or '')
    
    return {
        'story': story,
        'mod_story': mod_story,
        'original_lines': original_lines,
        'modified_lines': modified_lines,
        'initial_state': str(initial_state),
        'modified_state': str(modified_state),
        'assignment_id': str(assignment_id),
        'title': str(title),
        'modified_line_indices': modified_line_indices,  # Which lines were modified (e.g., [2, 4])
        'line_modifications': line_modifications  # Detailed modifications with before/after
    }


def process_dataset(client, data, dataset_name):
    """Process all samples in a dataset"""
    print(f"\n{'='*70}")
    print(f"Processing {dataset_name} dataset ({len(data)} samples)")
    print(f"{'='*70}\n")
    
    results = []
    counterfacts_only = []
    failed_count = 0
    
    for idx, item in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
        extracted = extract_story_and_states(item)
        
        # Skip if essential fields are missing
        if not extracted['initial_state'] or not extracted['modified_state']:
            failed_count += 1
            continue
        
        # Run inference
        result = test_fact_induction(
            client,
            extracted['initial_state'],
            extracted['modified_state'],
            extracted['story'],
            extracted['mod_story']
        )
        
        if result:
            # Add metadata to result
            result['assignment_id'] = extracted['assignment_id']
            result['title'] = extracted['title']
            result['dataset'] = dataset_name
            result['original_story_lines'] = extracted['original_lines']
            result['modified_story_lines'] = extracted['modified_lines']
            result['modified_line_indices'] = extracted['modified_line_indices']
            result['line_modifications'] = extracted['line_modifications']
            results.append(result)
            
            # Extract just the counterfact for the separate list
            if 'counterfact' in result:
                counterfacts_only.append({
                    'counterfact': result['counterfact'],
                    'state': result.get('state', ''),
                    'modify_state': result.get('modify_state', ''),
                    'assignment_id': result['assignment_id'],
                    'title': result['title'],
                    'dataset': dataset_name,
                    'modified_line_indices': extracted['modified_line_indices'],
                    'line_modifications': extracted['line_modifications']
                })
        else:
            failed_count += 1
    
    print(f"\n{dataset_name} Summary:")
    print(f"  Total samples: {len(data)}")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {failed_count}")
    
    return results, counterfacts_only


def summarize_counterfacts(client, counterfacts_list):
    """Use LLM to create a summarized global list of counterfacts"""
    print(f"\n{'='*70}")
    print("Creating summarized global list of counterfacts...")
    print(f"{'='*70}\n")
    
    # Extract just the counterfact texts
    counterfact_texts = [cf['counterfact'] for cf in counterfacts_list]
    
    # Group similar counterfacts (simple approach - can be enhanced)
    print("Analyzing counterfacts for patterns and similarities...")
    
    # Create prompt for summarization
    sample_counterfacts = counterfact_texts[:100]  # Use first 100 as sample
    
    prompt = f"""You are an expert at analyzing and categorizing fictional counterfactual statements.

I have a large collection of counterfactual (false, fictional) global facts. Here's a sample of {len(sample_counterfacts)} counterfacts:

{json.dumps(sample_counterfacts, indent=2)}

Please analyze these counterfacts and create a categorized summary that:
1. Identifies main themes/categories (e.g., language changes, food/diet changes, technology changes, social changes, etc.)
2. For each category, provide 2-3 representative examples
3. Estimate the distribution of themes across all counterfacts

Output in JSON format:
{{
    "categories": [
        {{
            "theme": "Category name",
            "description": "Brief description",
            "examples": ["example 1", "example 2"],
            "estimated_count": estimated number
        }}
    ],
    "total_analyzed": {len(sample_counterfacts)},
    "unique_patterns": number of unique patterns identified
}}"""

    try:
        response = client.chat.completions.create(
            model="qwen2.5-7b",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at analyzing and categorizing text data. You identify patterns and create clear summaries."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more focused analysis
            top_p=0.9,
            max_tokens=800,
        )
        
        result = response.choices[0].message.content
        
        # Parse JSON
        try:
            cleaned_result = result.strip()
            if cleaned_result.startswith("```json"):
                cleaned_result = cleaned_result[7:]
            if cleaned_result.endswith("```"):
                cleaned_result = cleaned_result[:-3]
            cleaned_result = cleaned_result.strip()
            
            summary = json.loads(cleaned_result)
            return summary
            
        except json.JSONDecodeError:
            print("Warning: Could not parse LLM summary response")
            return None
            
    except Exception as e:
        print(f"Error creating summary: {e}")
        return None


def create_simple_global_summary(counterfacts_list):
    """Create a simple statistical summary of counterfacts"""
    # Count unique counterfacts
    unique_counterfacts = {}
    for cf in counterfacts_list:
        text = cf['counterfact'].lower().strip()
        if text in unique_counterfacts:
            unique_counterfacts[text]['count'] += 1
            unique_counterfacts[text]['examples'].append({
                'assignment_id': cf['assignment_id'],
                'title': cf['title']
            })
        else:
            unique_counterfacts[text] = {
                'counterfact': cf['counterfact'],
                'count': 1,
                'examples': [{
                    'assignment_id': cf['assignment_id'],
                    'title': cf['title']
                }]
            }
    
    # Sort by frequency
    sorted_counterfacts = sorted(
        unique_counterfacts.values(),
        key=lambda x: x['count'],
        reverse=True
    )
    
    return {
        'total_counterfacts': len(counterfacts_list),
        'unique_counterfacts': len(unique_counterfacts),
        'most_common': sorted_counterfacts[:50],  # Top 50 most common
        'all_unique': sorted_counterfacts
    }


def main():
    """Main function"""
    print("="*70)
    print("Qwen2.5-7B with vLLM Server API - Counterfactual Analysis")
    print("Processing tr_data.jsonl, te_data.jsonl, and human_eval_dat.csv")
    print("="*70)
    
    # Initialize OpenAI client to connect to vLLM server
    VLLM_SERVER_URL = "http://localhost:22003/v1"
    
    print(f"\nConnecting to vLLM server at: {VLLM_SERVER_URL}")
    print("Make sure the vLLM server is running!")
    print("(Run: ./start_vllm_server_background.sh)\n")
    
    client = OpenAI(
        base_url=VLLM_SERVER_URL,
        api_key="dummy"  # vLLM doesn't require a real API key
    )
    
    # Test connection
    try:
        models = client.models.list()
        print("✓ Connected to vLLM server successfully!")
        print(f"Available models: {[model.id for model in models.data]}\n")
    except Exception as e:
        print(f"✗ Failed to connect to vLLM server: {e}")
        print("Please start the server first: ./start_vllm_server_background.sh")
        return
    
    # Define file paths
    base_path = "/shared/nas2/knguye71/conterfactual_dataset"
    tr_data_path = os.path.join(base_path, "data", "tr_data.jsonl")
    te_data_path = os.path.join(base_path, "data", "te_data.jsonl")
    human_eval_path = os.path.join(base_path, "data", "human_eval_dat.csv")
    
    # Check if files exist
    if not os.path.exists(tr_data_path):
        print(f"Error: Training data file not found: {tr_data_path}")
        return
    if not os.path.exists(te_data_path):
        print(f"Error: Test data file not found: {te_data_path}")
        return
    if not os.path.exists(human_eval_path):
        print(f"Error: Human eval data file not found: {human_eval_path}")
        return
    
    # Load datasets
    print("Loading datasets...")
    tr_data = load_jsonl(tr_data_path)
    te_data = load_jsonl(te_data_path)
    human_eval_data = load_csv_as_dict(human_eval_path)
    print(f"✓ Loaded {len(tr_data)} training samples")
    print(f"✓ Loaded {len(te_data)} test samples")
    print(f"✓ Loaded {len(human_eval_data)} human evaluation samples")
    
    # Process training data
    tr_results, tr_counterfacts = process_dataset(client, tr_data, "training")
    
    # Process test data
    te_results, te_counterfacts = process_dataset(client, te_data, "test")
    
    # Process human evaluation data
    human_eval_results, human_eval_counterfacts = process_dataset(client, human_eval_data, "human_eval")
    
    # Combine all counterfacts
    all_counterfacts = tr_counterfacts + te_counterfacts + human_eval_counterfacts
    
    # Save results
    print(f"\n{'='*70}")
    print("Saving results...")
    print(f"{'='*70}\n")
    
    # Save training results
    tr_output_path = os.path.join(base_path, "results", "counterfact_result_tr_qwen3.json")
    with open(tr_output_path, "w", encoding="utf-8") as f:
        json.dump(tr_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Training results saved to: {tr_output_path}")
    print(f"  Total: {len(tr_results)} samples")
    
    # Save test results
    te_output_path = os.path.join(base_path, "results", "counterfact_result_te_qwen3.json")
    with open(te_output_path, "w", encoding="utf-8") as f:
        json.dump(te_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Test results saved to: {te_output_path}")
    print(f"  Total: {len(te_results)} samples")
    
    # Save human eval results
    human_eval_output_path = os.path.join(base_path, "results", "counterfact_result_human_eval_qwen3.json")
    with open(human_eval_output_path, "w", encoding="utf-8") as f:
        json.dump(human_eval_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Human eval results saved to: {human_eval_output_path}")
    print(f"  Total: {len(human_eval_results)} samples")
    
    # Save combined results
    all_results = tr_results + te_results + human_eval_results
    all_output_path = os.path.join(base_path, "results", "counterfact_result_all_qwen3.json")
    with open(all_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"✓ Combined results saved to: {all_output_path}")
    print(f"  Total: {len(all_results)} samples")
    
    # Save counterfacts only list
    counterfacts_output_path = os.path.join(base_path, "results", "counterfacts_list_qwen3.json")
    with open(counterfacts_output_path, "w", encoding="utf-8") as f:
        json.dump(all_counterfacts, f, indent=2, ensure_ascii=False)
    print(f"✓ Counterfacts list saved to: {counterfacts_output_path}")
    print(f"  Total: {len(all_counterfacts)} counterfacts")
    
    # Save separate counterfacts lists for each dataset
    tr_counterfacts_path = os.path.join(base_path, "results", "counterfacts_list_tr_qwen3.json")
    with open(tr_counterfacts_path, "w", encoding="utf-8") as f:
        json.dump(tr_counterfacts, f, indent=2, ensure_ascii=False)
    print(f"✓ Training counterfacts list saved to: {tr_counterfacts_path}")
    
    te_counterfacts_path = os.path.join(base_path, "results", "counterfacts_list_te_qwen3.json")
    with open(te_counterfacts_path, "w", encoding="utf-8") as f:
        json.dump(te_counterfacts, f, indent=2, ensure_ascii=False)
    print(f"✓ Test counterfacts list saved to: {te_counterfacts_path}")
    
    human_eval_counterfacts_path = os.path.join(base_path, "results", "counterfacts_list_human_eval_qwen3.json")
    with open(human_eval_counterfacts_path, "w", encoding="utf-8") as f:
        json.dump(human_eval_counterfacts, f, indent=2, ensure_ascii=False)
    print(f"✓ Human eval counterfacts list saved to: {human_eval_counterfacts_path}")
    
    # Create simple statistical summary for each dataset
    print(f"\n{'='*70}")
    print("Creating statistical summaries for each dataset...")
    print(f"{'='*70}\n")
    
    # Training summary
    print("Creating training data summary...")
    tr_summary = create_simple_global_summary(tr_counterfacts)
    tr_summary_path = os.path.join(base_path, "results", "counterfacts_global_summary_tr_qwen3.json")
    with open(tr_summary_path, "w", encoding="utf-8") as f:
        json.dump(tr_summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Training summary saved to: {tr_summary_path}")
    print(f"  Total: {tr_summary['total_counterfacts']}, Unique: {tr_summary['unique_counterfacts']}")
    
    # Test summary
    print("\nCreating test data summary...")
    te_summary = create_simple_global_summary(te_counterfacts)
    te_summary_path = os.path.join(base_path, "results", "counterfacts_global_summary_te_qwen3.json")
    with open(te_summary_path, "w", encoding="utf-8") as f:
        json.dump(te_summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Test summary saved to: {te_summary_path}")
    print(f"  Total: {te_summary['total_counterfacts']}, Unique: {te_summary['unique_counterfacts']}")
    
    # Human eval summary
    print("\nCreating human eval data summary...")
    human_eval_summary = create_simple_global_summary(human_eval_counterfacts)
    human_eval_summary_path = os.path.join(base_path, "results", "counterfacts_global_summary_human_eval_qwen3.json")
    with open(human_eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(human_eval_summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Human eval summary saved to: {human_eval_summary_path}")
    print(f"  Total: {human_eval_summary['total_counterfacts']}, Unique: {human_eval_summary['unique_counterfacts']}")
    
    # Create combined statistical summary
    print("\nCreating combined statistical summary...")
    simple_summary = create_simple_global_summary(all_counterfacts)
    
    simple_summary_path = os.path.join(base_path, "results", "counterfacts_global_summary_all_qwen3.json")
    with open(simple_summary_path, "w", encoding="utf-8") as f:
        json.dump(simple_summary, f, indent=2, ensure_ascii=False)
    print(f"✓ Combined global summary saved to: {simple_summary_path}")
    print(f"  Total counterfacts: {simple_summary['total_counterfacts']}")
    print(f"  Unique counterfacts: {simple_summary['unique_counterfacts']}")
    
    # Create LLM-based categorical summary for combined data
    print(f"\n{'='*70}")
    print("Creating LLM-based categorical summary...")
    print(f"{'='*70}\n")
    llm_summary = summarize_counterfacts(client, all_counterfacts)
    
    if llm_summary:
        llm_summary_path = os.path.join(base_path, "results", "counterfacts_categorical_summary_all_qwen3.json")
        with open(llm_summary_path, "w", encoding="utf-8") as f:
            json.dump(llm_summary, f, indent=2, ensure_ascii=False)
        print(f"✓ Categorical summary saved to: {llm_summary_path}")
    
    print(f"\n{'='*70}")
    print("Processing complete!")
    print(f"{'='*70}")
    print("\nOutput files created:")
    print("\nFull Results:")
    print(f"  1. {tr_output_path}")
    print(f"  2. {te_output_path}")
    print(f"  3. {human_eval_output_path}")
    print(f"  4. {all_output_path}")
    print("\nCounterfacts Lists:")
    print(f"  5. {tr_counterfacts_path}")
    print(f"  6. {te_counterfacts_path}")
    print(f"  7. {human_eval_counterfacts_path}")
    print(f"  8. {counterfacts_output_path}")
    print("\nGlobal Summaries (Statistical):")
    print(f"  9. {tr_summary_path}")
    print(f"  10. {te_summary_path}")
    print(f"  11. {human_eval_summary_path}")
    print(f"  12. {simple_summary_path}")
    if llm_summary:
        print("\nCategorical Summary (LLM-based):")
        print(f"  13. {llm_summary_path}")


if __name__ == "__main__":
    main()
