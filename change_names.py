import os

root = "dataset1"

start_number = 70

for word in os.listdir(root):
    folder = os.path.join(root, word)

    if not os.path.isdir(folder):
        continue
    files = sorted([
        f for f in os.listdir(folder)
        if f.endswith(".npy")
    ])

    counter = start_number

    for file in files:
        old_path = os.path.join(folder, file)

        new_name = f"{counter:03d}.npy"
        new_path = os.path.join(folder,new_name)

        os.rename(old_path, new_path)
        counter +=1

    print(f"{word} : renamed {len(files)} files")