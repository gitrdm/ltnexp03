#!/usr/bin/env python3
"""
Persistence Layer Examples Summary

This script provides an overview of the comprehensive persistence layer
implementation and available demonstration scripts.

Available Examples:
===================

1. demo_persistence_layer.py - Complete feature demonstration
2. persistence_strategy_example.py - Strategy implementation showcase  
3. multi_format_persistence_example.py - Multi-format storage demonstration

Each script demonstrates different aspects of the persistence layer strategy
outlined in PERSISTENCE_LAYER_STRATEGY.md.
"""

import sys
import subprocess
from pathlib import Path


def print_header(title: str, emoji: str = "🚀"):
    """Print formatted header."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))


def print_feature(feature: str, description: str = ""):
    """Print feature with description."""
    if description:
        print(f"✅ {feature}: {description}")
    else:
        print(f"✅ {feature}")


def main():
    """Display persistence layer overview and run options."""
    print("🌟 Soft Logic Microservice - Persistence Layer Examples")
    print("=" * 65)
    
    print_header("Persistence Layer Strategy Overview", "📋")
    
    print("The persistence layer implements a hybrid approach optimized for:")
    features = [
        ("Batch Operations", "JSONL format for efficient append-only operations"),
        ("Complex Queries", "SQLite for joins, indexes, and transactions"),
        ("Vector Storage", "NPZ format for compressed embeddings"),
        ("Workflow Management", "Complete batch operation tracking"),
        ("Data Safety", "Soft deletes with compaction and backup"),
        ("Memory Efficiency", "Streaming operations for large datasets"),
        ("Contract Validation", "Design by Contract with icontract"),
        ("Multi-Format Optimization", "Right format for each use case")
    ]
    
    for feature, description in features:
        print_feature(feature, description)
    
    print_header("Available Demo Scripts", "🎬")
    
    scripts = [
        {
            "name": "demo/demo_persistence_layer.py",
            "title": "Complete Persistence Demo",
            "description": "Comprehensive demonstration of all persistence features",
            "highlights": [
                "Contract-enhanced persistence manager",
                "Batch workflow creation and processing", 
                "Streaming queries and filtering",
                "Soft deletes and compaction",
                "Performance testing",
                "Storage integrity validation"
            ]
        },
        {
            "name": "examples/persistence_strategy_example.py", 
            "title": "Strategy Implementation Showcase",
            "description": "Focused demonstration of strategy document recommendations",
            "highlights": [
                "JSONL batch operations strategy",
                "Incremental update patterns",
                "Workflow management features",
                "Performance characteristics",
                "Storage format analysis"
            ]
        },
        {
            "name": "examples/multi_format_persistence_example.py",
            "title": "Multi-Format Storage Demo", 
            "description": "Concrete examples of each storage format",
            "highlights": [
                "JSONL format for batch operations",
                "SQLite for complex queries",
                "NPZ for vector embeddings", 
                "Workflow file management",
                "Directory structure creation",
                "Format-specific benefits"
            ]
        }
    ]
    
    for i, script in enumerate(scripts, 1):
        print(f"\n{i}. {script['title']}")
        print(f"   📄 {script['name']}")
        print(f"   📝 {script['description']}")
        print("   🎯 Key features:")
        for highlight in script['highlights']:
            print(f"      • {highlight}")
    
    print_header("Quick Start", "⚡")
    
    print("To run any demo script:")
    print("   python <script_name>")
    print()
    print("Examples:")
    for script in scripts:
        print(f"   python {script['name']}")
    
    print_header("Implementation Files", "🏗️")
    
    implementation_files = [
        ("app/core/persistence.py", "Basic persistence manager"),
        ("app/core/batch_persistence.py", "Batch workflow manager"),
        ("app/core/contract_persistence.py", "Contract-enhanced manager"),
        ("app/core/protocols.py", "Persistence protocols/interfaces"),
        ("app/batch_service.py", "FastAPI service layer"),
        ("tests/test_core/test_persistence.py", "Comprehensive test suite")
    ]
    
    print("Core implementation files:")
    for file_path, description in implementation_files:
        status = "✅" if Path(file_path).exists() else "❌"
        print(f"   {status} {file_path} - {description}")
    
    print_header("Strategy Documentation", "📚")
    
    docs = [
        ("PERSISTENCE_LAYER_STRATEGY.md", "Complete strategy and recommendations"),
        ("PERSISTENCE_IMPLEMENTATION_STATUS.md", "Implementation status and validation")
    ]
    
    for doc, description in docs:
        status = "✅" if Path(doc).exists() else "❌"
        print(f"   {status} {doc} - {description}")
    
    print_header("Interactive Demo Selection", "🎮")
    
    print("Select a demo to run:")
    for i, script in enumerate(scripts, 1):
        print(f"   {i}. {script['title']}")
    print("   4. Exit")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "4":
            print("👋 Goodbye!")
            return
        
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(scripts):
                script_name = scripts[choice_idx]['name']
                print(f"\n🚀 Running {script_name}...")
                print("=" * 50)
                
                # Run the selected script
                result = subprocess.run([sys.executable, script_name], 
                                      capture_output=False, 
                                      text=True)
                
                if result.returncode == 0:
                    print(f"\n✅ {script_name} completed successfully!")
                else:
                    print(f"\n❌ {script_name} encountered an error")
                
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except ValueError:
            print("❌ Invalid input. Please enter a number 1-4.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo selection cancelled by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    """Run the persistence layer examples overview."""
    main()
