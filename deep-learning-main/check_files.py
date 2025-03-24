import os

base_dir = "data_source/Training"
classes = ["no_tumor", "pituitary_tumor", "meningioma_tumor", "glioma_tumor"]

for category in classes:
    path = os.path.join(base_dir, category)
    if not os.path.exists(path):
        print(f"Missing directory: {path}")
    else:
        images = [f for f in os.listdir(path) if f.endswith((".jpg", ".png", ".jpeg"))]
        print(f"Found {len(images)} images in {path}")
