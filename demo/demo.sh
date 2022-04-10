#!/bin/bash                                                                                           
# Add dependency into projects                                                                        
PROJ_DIR=$HOME/WORK/projects/deepsymbolic/code/
PYTHONPATH=$PYTHONPATH:$PROJ_DIR
export PYTHONPATH

streamlit run demo.py
