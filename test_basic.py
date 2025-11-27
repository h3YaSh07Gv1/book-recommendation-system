#!/usr/bin/env python3
"""
Basic functionality test for the NextGen Book Recommender.
This script tests core functionality without requiring full test suite setup.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test basic imports work."""
    try:
        from src.data.loader import data_loader
        from src.engine.recommender import Recommender
        from src.ui.dashboard import create_dashboard
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_loading():
    """Test data loading functionality."""
    try:
        from src.data.loader import data_loader
        # Test loading tagged descriptions
        docs = data_loader.load_documents("data/tagged_description.txt")
        if docs and len(docs) > 0:
            print(f"✅ Loaded {len(docs)} documents")
            return True
        else:
            print("❌ No documents loaded")
            return False
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_recommender_creation():
    """Test recommender initialization."""
    try:
        from src.data.loader import data_loader
        from src.engine.recommender import Recommender

        docs = data_loader.load_documents("data/tagged_description.txt")
        if not docs:
            print("❌ No documents available for recommender")
            return False

        recommender = Recommender(docs)
        print("✅ Recommender initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Recommender creation failed: {e}")
        return False

def test_basic_search():
    """Test basic search functionality."""
    try:
        from src.data.loader import data_loader
        from src.engine.recommender import Recommender

        docs = data_loader.load_documents("data/tagged_description.txt")
        if not docs:
            print("❌ No documents available for search")
            return False

        recommender = Recommender(docs)
        results = recommender.semantic_search("adventure story", k=5)

        if results and len(results) > 0:
            print(f"✅ Search returned {len(results)} results")
            return True
        else:
            print("❌ Search returned no results")
            return False
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 Running Basic Functionality Tests")
    print("=" * 50)

    tests = [
        ("Import Test", test_imports),
        ("Data Loading Test", test_data_loading),
        ("Recommender Creation Test", test_recommender_creation),
        ("Basic Search Test", test_basic_search),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All basic functionality tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
