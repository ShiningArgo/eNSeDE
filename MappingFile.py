import os, json
from Mapping import COORDINATES_MAP

# SAVE COORDINATES MAP - Serializes the coordinate mapping to a JSON file
def save_coordinates_map(coordinates_map: 'COORDINATES_MAP', maps_folder: str, video_filename: str) -> None:
    os.makedirs(maps_folder, exist_ok=True)

    filename = video_filename.split('.')[0]
    coordinates_map_filename = filename + '.json'

    # Clean the path for consistent slash formatting
    path = os.path.normpath(os.path.join(maps_folder, coordinates_map_filename))

    # Overwrite the existing map file if it is already present
    if os.path.isfile(path):
        print(f'Coordinates map file {path} already exists! Overwriting...')

    with open(path, 'w+') as file:
        file.write(
            json.dumps(
                coordinates_map,
                indent=True
            )
        )

        print(f'Coordinates map saved to {path}')


# LOAD COORDINATES MAP - Deserializes and returns the coordinate mapping from a JSON file associated with a specific video
def load_coordinates_map(maps_folder: str, video_filename: str) -> 'COORDINATES_MAP':
    filename = video_filename.split('.')[0]
    coordinates_map_filename = filename + '.json'
    path = os.path.normpath(os.path.join(maps_folder, coordinates_map_filename))

    if not os.path.isfile(path):
        raise IOError(f'Coordinates map file {path} doesn\'t exist!')

    with open(path, 'r') as file:
        return json.load(file)