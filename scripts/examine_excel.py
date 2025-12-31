import pandas as pd
import os

# Change to the correct directory
os.chdir(r"C:\Users\Thomas\OneDrive\Desktop\Parsing")

# Read the expected output file
df = pd.read_excel('structured (6).xlsx')

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst 3 rows:")
print(df.head(3).to_string())

# Check for non-null values in key columns
print("\nNon-null counts:")
for col in df.columns:
    non_null = df[col].notna().sum()
    print(f"{col}: {non_null}/{len(df)}")

# Show sample of non-null values
print("\nSample non-null values from Title Number:")
title_samples = df[df['Title Number'].notna()]['Title Number'].head(3)
print(title_samples.tolist())

print("\nSample non-null values from Chapter Number:")
chapter_samples = df[df['Chapter Number'].notna()]['Chapter Number'].head(3)
print(chapter_samples.tolist())

print("\nSample non-null values from Section Number:")
section_samples = df[df['Section Number'].notna()]['Section Number'].head(3)
print(section_samples.tolist()) 