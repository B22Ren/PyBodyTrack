# PyBodyTrack

pip install numpy pandas matplotlib mediapipe opencv-python
pip install ffmpeg-python
python animate_pose_csv.py --csv pose3d_mono.csv --out animation.mp4 --fps 30

All codes require a webcam.

Monocular Pose Estimation (monocular_pose3d.py with animate_pose_csv.py
):
Works with single-camera pose data to reconstruct 3D positions.A script that stores the information into an csv and another script that reconstructs it in an 3D array, where the landmarks form a stick figure skeleton animating it in a 3d spaced cube

Air Keyboard (air_keyboard.py):
Control a virtual keyboard with hand movements and u can search online on multiple platforms like instagram,youtube and google.

Pet Interaction (pet_interaction.py):
Interact with a virtual pet using body/hand gestures. You can pet him, offfer a treat or a toy (ball) he goes after.

Arrow Shooting Demo (Arrowshooting.py):
Simulates archery-like gestures to shoot arrows.You need to use two hands in which one serves as an indicator for the direction and the strenght used.

Energy Ball Effect (energy_ball.py):
Creates a fun animation where you can charge and throw an energy ball.

Painting with Gestures (painting.py):
Draw in virtual space using hand motions. Where you can change colors,thickness and also rotate the object, it is drawn in a 3d space.

Portals Demo (portals.py):
Generates portal-like effects, the portals look like circles and their dimension can be increased and also moved around.

Orchestration Demo (Dirijor.py):
A “conductor” style demo where hand motions can control some sounds. for it u need a special library if it doesnt work it offers a beeping sound in order to show its functionality.
