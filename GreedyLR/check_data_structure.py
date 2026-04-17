import json

# Load and inspect the data structure
with open('robust_results.json', 'r') as f:
    data = json.load(f)

print("Data type:", type(data))
print("Length:", len(data))

if isinstance(data, list) and len(data) > 0:
    print("\nFirst experiment structure:")
    first_exp = data[0]
    print("Keys:", list(first_exp.keys()) if isinstance(first_exp, dict) else "Not a dict")

    if 'loss_history' in first_exp:
        print("Loss history length:", len(first_exp['loss_history']))
        print("First few losses:", first_exp['loss_history'][:5])

    print("\nFull first experiment:")
    print(json.dumps(first_exp, indent=2)[:500] + "...")

elif isinstance(data, dict):
    print("\nFirst few keys:", list(data.keys())[:5])
    first_key = list(data.keys())[0]
    first_exp = data[first_key]
    print("First experiment structure:")
    print("Keys:", list(first_exp.keys()) if isinstance(first_exp, dict) else "Not a dict")