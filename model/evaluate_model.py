
import os
from predict import predict_retinopathy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")

classes = [
    "No_DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative_DR",
]

total = 0
correct = 0

print("\nDiabetic Retinopathy Model Evaluation")
print("-" * 45)
print("Test directory:", TEST_DIR)

for actual_class in classes:
    class_dir = os.path.join(TEST_DIR, actual_class)

    if not os.path.isdir(class_dir):
        print("Folder not found:", class_dir)
        continue

    files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    print(f"\nChecking {actual_class}: {len(files)} images")

    for filename in files:
        image_path = os.path.join(class_dir, filename)
        result = predict_retinopathy(image_path)

        if result["status"] != "SUCCESS":
            print(f"Error: {filename}: {result['error']}")
            continue

        predicted_class = result["predicted_class"]
        is_correct = predicted_class == actual_class

        total += 1
        if is_correct:
            correct += 1

        status = "Correct" if is_correct else "Incorrect"
        print(
            f"{filename}: Actual={actual_class}, "
            f"Predicted={predicted_class}, {status}"
        )

print("\n" + "-" * 45)
print("Total evaluated:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)

if total > 0:
    accuracy = correct / total * 100
    print(f"Test accuracy: {accuracy:.2f}%")
else:
    print("No images were evaluated. Check the paths and model.")
