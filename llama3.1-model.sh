#!/bin/bash

# Variables
model_name = "llama3.1"
custom_model_name = "llama-crew"

# get the abse model
ollama pull $model_name

# Create the model file
ollama create $custom_model_name -f ./