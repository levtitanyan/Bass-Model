import os

# Define the notebook filename
notebook_file = "HW1.ipynb"  # Change this to your actual notebook filename

# Convert the notebook to PDF
os.system(f"jupyter nbconvert --to pdf {notebook_file}")

print(f"{notebook_file} has been successfully converted to PDF!")
