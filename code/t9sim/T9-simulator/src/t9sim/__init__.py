"""T9 synthetic ad-auction simulator - clean-room rebuild.

A real installable package (no sys.path hacks): config/, calibration/ and
output/ are co-located under the package root and resolved through paths.py.
The column/feature schema is centralised in schema.py (the single source of
truth), with the test suite holding an independent hardcoded copy as the oracle.
"""
__version__ = "0.1.0"
