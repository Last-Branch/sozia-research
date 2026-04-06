"""Allow ``python -m tsl_recognition <command>`` from the project root.

    python -m tsl_recognition extract --test
    python -m tsl_recognition train
    python -m tsl_recognition infer --mode motion
    python -m tsl_recognition validate
"""
from .cli import main

main()
