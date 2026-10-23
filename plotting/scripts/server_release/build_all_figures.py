#!/usr/bin/env python3
"""Rebuild the manuscript figures from the checked-in source tables.

Main-text Figures 2--7 are built by ``../rebuild_main_figures.py`` from
``plotting/data/derived/main``, ``plotting/data/load`` and
``plotting/data/server_checked``.  Figure 1 is the conceptual framework image
preserved in ``plotting/figures/main/``; Figure 8 is typeset from
``table2a``/``table2b`` in the manuscript's unified layout pass.

This module rebuilds the E-Companion figures from
``plotting/data/derived/ec``.
"""
from figure_builders import ec_r1, ec_r3, ec_r4, ec_r5, ec_r6, ec_r8, ec_r9

BUILDERS = [ec_r1, ec_r3, ec_r4, ec_r5, ec_r6, ec_r8, ec_r9]

if __name__ == "__main__":
    for fn in BUILDERS:
        print(f"[figure] {fn.__name__}")
        fn()
