"""
conftest.py — Pytest configuration for the test suite.
"""
import sys
import os

# Add project root to path so tests can import src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
