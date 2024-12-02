import inflect

# Initialize the inflect engine
p = inflect.engine()

# Function to generate text-numeric pairs
def generate_dataset(start, end):
    data = []
    for num in range(start, end + 1):
        # Convert number to text
        text = p.number_to_words(num)
        # Clean text (remove commas, standardize case, etc.)
        text = text.replace(",", "").lower()
        # Append pair to dataset
        data.append((text, str(num)))
    return data

# Generate dataset for numbers 0 to 9999
dataset = generate_dataset(0, 9999)

# Save dataset to a file
with open("text_to_number_dataset.csv", "w") as f:
    f.write("text,number\n")
    for text, num in dataset:
        f.write(f"{text},{num}\n")
