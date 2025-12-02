"""Dataset management for LangSmith evaluations.

Provides utilities for creating and managing LangSmith datasets from course JSON files.

Usage:
    # Create a dataset from course outputs (auto-generates name from course title)
    python -m evaluation.dataset create-dataset --inputs output/*.json
    
    # Create a dataset with custom name
    python -m evaluation.dataset create-dataset --inputs output/*.json --name my-courses
    
    # List all available datasets
    python -m evaluation.dataset list-datasets
"""

import json
import os
from typing import List, Optional

from langsmith import Client

from main.state import CourseState


# ---- Utility Functions ----

def load_course_state(input_file: str) -> CourseState:
    """Load CourseState from JSON file."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return CourseState.model_validate(data)


def get_client() -> Client:
    """Get LangSmith client."""
    return Client()


# ---- Dataset Management ----

def create_dataset(
    input_files: List[str],
    dataset_name: Optional[str] = None,
    description: str = "Course generation outputs for evaluation",
    use_existing: bool = False,
) -> tuple[str, str, List[str]]:
    """
    Create a LangSmith dataset from course JSON files.
    
    Args:
        input_files: List of paths to course JSON files
        dataset_name: Name for the dataset (optional - auto-generated from course title if not provided)
        description: Dataset description
        use_existing: If True, add examples to existing dataset; if False, error on conflict
        
    Returns:
        Tuple of (dataset_id, dataset_name, example_ids)
    """
    client = get_client()
    
    # Auto-generate dataset name from course title if not provided
    if dataset_name is None:
        if not input_files:
            raise ValueError("No input files provided")
        
        # Load first file to extract title
        first_file = input_files[0]
        if not os.path.exists(first_file):
            raise FileNotFoundError(f"File not found: {first_file}")
        
        course_state = load_course_state(first_file)
        # Sanitize title for dataset name: lowercase, replace spaces/special chars with hyphens
        sanitized_title = course_state.title.lower()
        sanitized_title = ''.join(c if c.isalnum() or c.isspace() else '-' for c in sanitized_title)
        dataset_name = '-'.join(sanitized_title.split())
        
        print(f"📝 Auto-generated dataset name from course title: {dataset_name}")
    
    existing_datasets = {ds.name: ds for ds in client.list_datasets()}
    if dataset_name in existing_datasets:
        dataset = existing_datasets[dataset_name]
        print(f"📦 Using existing dataset: {dataset_name} (ID: {dataset.id})")
    else:
        # Create new dataset
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description,
        )
        print(f"📦 Created dataset: {dataset_name} (ID: {dataset.id})")
    
    # Fetch existing examples to check for duplicates
    existing_examples = list(client.list_examples(dataset_id=dataset.id))
    existing_file_paths = {ex.inputs.get("file_path"): str(ex.id) for ex in existing_examples}
    
    # Add examples and track their IDs
    example_ids = []
    added_count = 0
    skipped_count = 0
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"   ⚠ Skipping {file_path} - file not found")
            continue
        
        # Check if example already exists
        if file_path in existing_file_paths:
            example_id = existing_file_paths[file_path]
            example_ids.append(example_id)
            course_state = load_course_state(file_path)
            print(f"   ⏭ Skipped: {course_state.title} (already in dataset)")
            skipped_count += 1
            continue
            
        try:
            course_state = load_course_state(file_path)
            
            # Store course data as the example input
            example = client.create_example(
                inputs={
                    "file_path": file_path,
                    "course_title": course_state.title,
                    "course_data": course_state.model_dump(),
                },
                outputs={},  # No ground truth for generative tasks
                dataset_id=dataset.id,
                metadata={
                    "num_modules": len(course_state.modules),
                    "language": course_state.language,
                }
            )
            example_ids.append(str(example.id))
            print(f"   ✓ Added: {course_state.title}")
            added_count += 1
        except Exception as e:
            print(f"   ✗ Failed to add {file_path}: {e}")
    
    if added_count > 0 or skipped_count > 0:
        print(f"\n✅ Dataset ready: {added_count} added, {skipped_count} skipped")
    
    return str(dataset.id), dataset_name, example_ids


def list_datasets():
    """List all available datasets."""
    client = get_client()
    datasets = list(client.list_datasets())
    
    print("\n📚 Available Datasets:")
    print("-" * 60)
    for ds in datasets:
        print(f"  {ds.name}")
        print(f"    ID: {ds.id}")
        print(f"    Examples: {ds.example_count}")
        print(f"    Created: {ds.created_at}")
        print()


# ---- CLI ----

def main():
    """Command line interface for dataset management."""
    import argparse
    import glob
    
    parser = argparse.ArgumentParser(description="LangSmith Dataset Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create dataset command
    create_parser = subparsers.add_parser("create-dataset", help="Create a LangSmith dataset")
    create_parser.add_argument("--inputs", type=str, required=True, help="Glob pattern for input JSON files")
    create_parser.add_argument("--name", type=str, default=None, help="Dataset name (optional - auto-generated from course title if not provided)")
    create_parser.add_argument("--description", type=str, default="Course generation outputs", help="Dataset description")
    
    # List datasets command
    subparsers.add_parser("list-datasets", help="List available datasets")
    
    args = parser.parse_args()
    
    if args.command == "create-dataset":
        input_files = glob.glob(args.inputs)
        if not input_files:
            print(f"No files found matching: {args.inputs}")
            return
        dataset_id, dataset_name, example_ids = create_dataset(
            input_files, 
            args.name, 
            args.description,
        )
        print(f"\n💡 Use this dataset name for evaluation: {dataset_name}")
        print(f"   Dataset contains {len(example_ids)} example(s)")
        
    elif args.command == "list-datasets":
        list_datasets()
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

