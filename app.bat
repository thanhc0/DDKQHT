@echo off
cd /F F:\Projects\Du doan KQ hoc tap\Final
start /B cmd /C "python -m streamlit run main.py & timeout /T 1 > nul"