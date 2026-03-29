#!/bin/bash

saved_model_dir=$1
saved_model_cli show --dir ${saved_model_dir} --all
