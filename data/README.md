# Data Directory

This directory keeps small label metadata in git and leaves raw WSIs plus extracted H5 embeddings out of git.

Put raw WSIs under `data/raw/`, generate AtlasPatch H5 files under `data/features/`, or point the environment variables in `configs/paths.example.env` to external storage.
