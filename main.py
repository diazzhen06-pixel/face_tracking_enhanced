"""Root entry point for the face tracking surveillance app."""

import runpy


if __name__ == "__main__":
    runpy.run_module("src.face_tracking.app", run_name="__main__")
