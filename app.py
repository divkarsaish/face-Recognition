import os
import face_recognition
import shutil
from sklearn.cluster import DBSCAN
import numpy as np
from tkinter import filedialog, Tk

def select_new_image():
    root = Tk()
    root.withdraw()
    new_image_path = filedialog.askopenfilename(title="Select New Image")
    return new_image_path

# Load the new photo from the selected path
new_image_path = select_new_image()
new_image = face_recognition.load_image_file(new_image_path)
new_face_locations = face_recognition.face_locations(new_image)
new_face_encodings = face_recognition.face_encodings(new_image, new_face_locations)

# Rest of your existing code...
known_faces = []
known_face_encodings = []
for filename in os.listdir("Pictures"):
    if filename.endswith(".jpg") and filename != "new_photo.jpg":
        image = face_recognition.load_image_file(os.path.join("Pictures", filename))
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if len(face_encodings) > 0:
            known_faces.append(filename)
            known_face_encodings.append(face_encodings[0])

print(f"Number of known faces: {len(known_faces)}")

# Combine new and known face encodings
all_face_encodings = np.vstack([np.array(known_face_encodings), np.array(new_face_encodings)])

# Use DBSCAN for clustering
dbscan = DBSCAN(eps=0.5, min_samples=1)
labels = dbscan.fit_predict(all_face_encodings)

print(f"DBSCAN Labels: {labels}")

# Create a dictionary to store images for each cluster
clustered_images = {label: [] for label in set(labels)}

# Assign each image to its respective cluster
for label, filename in zip(labels[:-1], known_faces):
    clustered_images[label].append(filename)

print("Clustered Images:")
for label, image_list in clustered_images.items():
    print(f"Cluster {label}: {image_list}")

# Create group folders for each cluster and copy images
copied_photos = set()

for label, image_list in clustered_images.items():
    group_folder = f"groups/group_{label}"
    os.makedirs(group_folder, exist_ok=True)
    
    for filename in image_list:
        if filename not in copied_photos:
            shutil.copy(os.path.join("Pictures", filename), group_folder)
            copied_photos.add(filename)
            print(f"Copied {filename} to {group_folder}")

# Copy new photo to each group folder
for label in clustered_images.keys():
    group_folder = f"groups/group_{label}"
    shutil.copy(new_image_path, group_folder)
    print(f"Copied new photo to {group_folder}")

# Display results using Tkinter messagebox
from tkinter import messagebox

messagebox.showinfo("Process Completed", "Group folders created with similar photos, avoiding duplicates.")
