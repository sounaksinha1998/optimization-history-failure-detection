"""Generate publication figures.

Correctness-memory probe plots (3--7): publication/generate_correctness_figures.py
Legacy residual-memory pipeline: research/associative_memory_fft/probe_experiments/probe_analysis.py
"""

if __name__ == "__main__":
    import runpy
    from pathlib import Path

    runpy.run_path(Path(__file__).with_name("generate_correctness_figures.py"), run_name="__main__")
