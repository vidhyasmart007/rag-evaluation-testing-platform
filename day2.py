import json


def evaluate_answer(expected, actual):
    expected = expected.strip().lower()
    actual = actual.strip().lower()

    if expected == actual:
        return "PASS"

    return "FAIL"


with open("data/eval_dataset.json", "r", encoding="utf-8") as file:
    dataset = json.load(file)


passed = 0
failed = 0

for case in dataset:
    result = evaluate_answer(
        case["expected_answer"],
        case["actual_answer"]
    )

    print("Test ID:", case["id"])
    print("Question:", case["question"])
    print("Evaluation:", result)
    print("-" * 40)

    if result == "PASS":
        passed += 1
    else:
        failed += 1


total = len(dataset)

print("\n===== EVALUATION SUMMARY =====")
print("Total test cases:", total)
print("Passed:", passed)
print("Failed:", failed)

if total > 0:
    pass_rate = passed / total * 100
    print(f"Pass rate: {pass_rate:.2f}%")
else:
    print("No test cases available.")