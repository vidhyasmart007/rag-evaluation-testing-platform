def evaluate_answer(expected, actual):
    if expected.strip().lower() == actual.strip().lower():
        return "PASS"

    return "FAIL"


test_cases = [
    {
        "question": "How many annual leave days are provided?",
        "expected": "20 days",
        "actual": "20 days"
    },
    {
        "question": "How many annual leave days are provided?",
        "expected": "20 days",
        "actual": "10 days"
    },
    {
        "question": "Where is the office?",
        "expected": "Bengaluru",
        "actual": "bengaluru"
    }
]

passed = 0
failed = 0

for case in test_cases:
    result = evaluate_answer(
        case["expected"],
        case["actual"]
    )

    print("Question:", case["question"])
    print("Expected:", case["expected"])
    print("Actual:", case["actual"])
    print("Result:", result)
    print("-" * 30)

    if result == "PASS":
        passed += 1
    else:
        failed += 1

total = len(test_cases)

print("Total test cases:", total)
print("Passed:", passed)
print("Failed:", failed)
print("Pass rate:", f"{passed / total:.0%}")