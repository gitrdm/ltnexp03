# NLTK FrameNet Tutorial
# This script provides a hands-on introduction to using the FrameNet corpus
# with the Natural Language Toolkit (NLTK) in Python.

# --------------------------------------------------------------------------
# Section 1: Getting Started
# --------------------------------------------------------------------------

# Before you begin, you need to have NLTK installed and the FrameNet corpus
# downloaded.

# Step 1: Install NLTK (if you haven't already)
# Open your terminal or command prompt and run:
# pip install nltk

# Step 2: Download the FrameNet corpus using the NLTK downloader.
# Run the following commands in a Python interpreter.
import nltk

try:
    # Attempt to import the FrameNet corpus to see if it's available
    from nltk.corpus import framenet as fn
    print("FrameNet corpus is already available.")
except LookupError:
    print("FrameNet corpus not found. Downloading...")
    nltk.download('framenet_v17')
    from nltk.corpus import framenet as fn
    print("Download complete.")


# --------------------------------------------------------------------------
# Section 2: Exploring Frames
# --------------------------------------------------------------------------
# A semantic frame represents a concept or a situation (e.g., "Commerce_buy").
# Let's explore the frames available in FrameNet.

print("\n--- Exploring Frames ---")

# Get the total number of frames in the database
total_frames = len(fn.frames())
print(f"Total number of frames: {total_frames}")

# You can search for frames using a regular expression.
# The '(?i)' flag makes the search case-insensitive.
print("\nSearching for frames related to 'Medical':")
matching_frames = fn.frames(r'(?i)medical')
for frame in matching_frames[:5]: # Displaying first 5 for brevity
    print(f"- Found frame: '{frame.name}' (ID: {frame.ID})")

# You can retrieve a specific frame by its ID to get more details,
# such as its definition and its Frame Elements (FEs).
# Note: Frame ID 198 corresponds to the 'Intentionally_act' frame.
print("\nDetails for a specific frame (ID: 198 - 'Intentionally_act'):")
f = fn.frame(198)

print(f"Frame: {f.name}")
print(f"Definition: {f.definition}")

print("\nFrame Elements (Semantic Roles):")
# f.FE is a dictionary of Frame Elements for this frame
for fe_name, fe_obj in f.FE.items():
    print(f"- {fe_name}")

print("\nSome Lexical Units that evoke this frame:")
# f.lexUnit is a dictionary of Lexical Units for this frame
# We'll display the first 5 for brevity.
lu_names = list(f.lexUnit.keys())
for lu_name in lu_names[:5]:
    print(f"- {lu_name}")


# --------------------------------------------------------------------------
# Section 3: Working with Lexical Units (LUs)
# --------------------------------------------------------------------------
# Lexical Units (LUs) are the words that evoke a frame. They are typically
# a word paired with a part-of-speech (e.g., 'buy.v').

print("\n--- Working with Lexical Units (LUs) ---")

# Get the total number of lexical units
total_lus = len(fn.lus())
print(f"Total number of LUs: {total_lus}")

# Search for LUs using a regular expression
print("\nSearching for LUs related to 'doctor':")
matching_lus = fn.lus(r'(?i)doctor')
for lu in matching_lus:
    print(f"- LU: '{lu.name}', evokes Frame: '{lu.frame.name}'")

# Get more details about a specific LU by its ID.
# Note: LU ID 1621 corresponds to 'recluse.n'.
print("\nDetails for a specific LU (ID: 1621 - 'recluse.n'):")
lu = fn.lu(1621)

print(f"LU Name: {lu.name}")
print(f"Definition: {lu.definition}")
print(f"Evoked Frame: {lu.frame.name}")
print(f"Part of Speech: {lu.POS}")


# --------------------------------------------------------------------------
# Section 4: Accessing Annotated Sentences
# --------------------------------------------------------------------------
# FrameNet includes sentences annotated with frames and frame elements.
# These are incredibly useful for seeing how frames are used in context.

print("\n--- Accessing Annotated Sentences ---")

# Let's look at an LU with richly annotated examples.
# Instead of using a hardcoded ID which can be unstable, we'll find the LU programmatically.
# We'll use the 'buy.v' LU from the 'Commerce_buy' frame.
buy_lus = fn.lus(r'buy.v')
# Find the specific LU that belongs to the 'Commerce_buy' frame
lu = None
for l in buy_lus:
    if l.frame.name == 'Commerce_buy':
        lu = l
        break

if lu:
    print(f"\nExample sentences for LU: '{lu.name}' in Frame: '{lu.frame.name}'")

    # The 'exemplars' attribute contains annotated sentences.
    for i, sentence in enumerate(lu.exemplars[:3]): # Displaying first 3 for brevity
        print(f"\nExample {i+1}:")
        print(f"Text: {sentence.text}")
        
        # The annotation is stored in layers. The 'Target' layer shows the
        # word that evokes the frame. The annotation objects are tuples of (start, end).
        target_layer = sentence.Target
        if target_layer:
            start_index = target_layer[0][0]
            end_index = target_layer[0][1]
            target_word = sentence.text[start_index:end_index+1]
            print(f"Target (evoking word): '{target_word}'")

        # The 'FE' layer shows the annotated Frame Elements.
        fe_layer = sentence.FE
        if fe_layer and isinstance(fe_layer, dict):
            print("Annotated Frame Elements:")
            for fe_name, fe_annotations in fe_layer.items():
                for annotation in fe_annotations:
                    start_index = annotation[0]
                    end_index = annotation[1]
                    fe_text = sentence.text[start_index:end_index+1]
                    print(f"- {fe_name}: '{fe_text}'")
        else:
            print("No annotated Frame Elements.")
else:
    print("Could not find the 'buy.v' LU in the 'Commerce_buy' frame.")
