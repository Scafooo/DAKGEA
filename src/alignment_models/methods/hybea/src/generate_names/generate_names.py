from .name_analysis import run_name_analysis
from .prioritize_names import run_prioritize


def run_generate_names():
    run_name_analysis()
    run_prioritize()