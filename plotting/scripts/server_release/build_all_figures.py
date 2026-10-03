#!/usr/bin/env python3
from figure_builders import *

BUILDERS = [main_fig1, main_fig2, main_fig3, main_fig4, main_fig5, ec_r1, ec_r2, ec_r3, ec_r4, ec_r5, ec_r6, ec_r7, ec_r8, ec_r9, ec_r10]

if __name__ == "__main__":
    for fn in BUILDERS:
        print(f"[figure] {fn.__name__}")
        fn()
