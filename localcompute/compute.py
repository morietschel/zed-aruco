import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
import collections
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Transform:
    translation: np.ndarray  # [x, y, z]
    rotation: np.ndarray    # [x, y, z, w] quaternion

    @staticmethod
    def inverse(transform):
        # Convert quaternion to rotation matrix
        R = Rotation.from_quat([*transform.rotation[:3], transform.rotation[3]]).as_matrix()
        t = transform.translation
        
        # Compute inverse transformation
        R_inv = R.T
        t_inv = -R_inv @ t
        
        # Convert back to quaternion
        q_inv = Rotation.from_matrix(R_inv).as_quat()
        return Transform(t_inv, np.array([*q_inv[:3], q_inv[3]]))

    def __mul__(self, other):
        # Convert quaternions to rotation matrices
        R1 = Rotation.from_quat([*self.rotation[:3], self.rotation[3]]).as_matrix()
        R2 = Rotation.from_quat([*other.rotation[:3], other.rotation[3]]).as_matrix()
        
        # Compute combined rotation and translation
        R = R1 @ R2
        t = R1 @ other.translation + self.translation
        
        # Convert back to quaternion
        q = Rotation.from_matrix(R).as_quat()
        return Transform(t, np.array([*q[:3], q[3]]))

def interpolate_transforms(t1: Transform, t2: Transform, alpha: float) -> Transform:
    # Interpolate translation
    trans = t1.translation * (1-alpha) + t2.translation * alpha
    
    # Interpolate rotation using SLERP
    r1 = Rotation.from_quat([*t1.rotation[:3], t1.rotation[3]])
    r2 = Rotation.from_quat([*t2.rotation[:3], t2.rotation[3]])
    r = Rotation.from_rotvec(alpha * r2.as_rotvec() + (1 - alpha) * r1.as_rotvec())  # Linear interpolation in rotation vector space
    quat = r.as_quat()
    
    return Transform(trans, np.array([*quat[:3], quat[3]]))

def process_recording(recording_file: str, min_confidence: float = 0.0001) -> Dict[Tuple[int, int], Dict]:
    # Read the recording file directly into a DataFrame, filtering low confidence detections immediately
    df = pd.read_csv(recording_file, 
                     comment='#',
                     names=['marker_id', 'timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw', 'confidence'])
    
    # Filter low confidence detections immediately
    df = df[df['confidence'] >= min_confidence]
    
    # Convert to structured numpy arrays for faster processing
    transforms = np.column_stack([
        df['tx'], df['ty'], df['tz'],
        df['qx'], df['qy'], df['qz'], df['qw']
    ])
    
    # Create a more efficient data structure for processing
    observations = {
        'marker_ids': df['marker_id'].values,
        'timestamps': df['timestamp'].values,
        'transforms': transforms,
        'confidences': df['confidence'].values
    }
    
    # Process in batches by timestamp
    unique_timestamps = np.unique(observations['timestamps'])
    marker_relations = {}
    
    for timestamp in unique_timestamps:
        # Get all observations for this timestamp
        mask = observations['timestamps'] == timestamp
        timestamp_markers = observations['marker_ids'][mask]
        timestamp_transforms = observations['transforms'][mask]
        timestamp_confidences = observations['confidences'][mask]
        
        # Skip if less than 2 markers
        if len(timestamp_markers) < 2:
            continue
            
        # Process unique pairs efficiently
        marker_pairs = np.array(np.meshgrid(timestamp_markers, timestamp_markers)).T.reshape(-1, 2)
        # Filter to keep only pairs where first ID < second ID
        valid_pairs = marker_pairs[:, 0] < marker_pairs[:, 1]
        marker_pairs = marker_pairs[valid_pairs]
        
        for m1_idx, m2_idx in marker_pairs:
            # Get indices in the timestamp arrays
            idx1 = np.where(timestamp_markers == m1_idx)[0][0]
            idx2 = np.where(timestamp_markers == m2_idx)[0][0]
            
            # Calculate relative transform
            t1 = Transform(timestamp_transforms[idx1, :3], timestamp_transforms[idx1, 3:])
            t2 = Transform(timestamp_transforms[idx2, :3], timestamp_transforms[idx2, 3:])
            relative_transform = Transform.inverse(t1) * t2
            
            relation_key = (int(m1_idx), int(m2_idx))
            avg_confidence = (timestamp_confidences[idx1] + timestamp_confidences[idx2]) / 2
            
            if relation_key not in marker_relations:
                marker_relations[relation_key] = {
                    'marker1_id': int(m1_idx),
                    'marker2_id': int(m2_idx),
                    'relative_transform': relative_transform,
                    'confidence': avg_confidence,
                    'observation_count': 1
                }
            else:
                relation = marker_relations[relation_key]
                total_confidence = relation['confidence'] * relation['observation_count']
                alpha = avg_confidence / (total_confidence + avg_confidence)
                
                relation['relative_transform'] = interpolate_transforms(
                    relation['relative_transform'],
                    relative_transform,
                    alpha
                )
                relation['confidence'] = (relation['confidence'] * relation['observation_count'] + avg_confidence) / (relation['observation_count'] + 1)
                relation['observation_count'] += 1
    
    return marker_relations

def save_marker_relations(relations: Dict[Tuple[int, int], Dict], output_file: str):
    """
    Save marker relations to a CSV file.
    
    Args:
        relations: Dictionary of marker relations
        output_file: Path to save the output CSV file
    """
    with open(output_file, 'w') as f:
        f.write("# Marker Relations File\n")
        f.write("# Format: marker1_id,marker2_id,tx,ty,tz,qx,qy,qz,qw,confidence,observations\n")
        
        for relation in relations.values():
            trans = relation['relative_transform'].translation
            rot = relation['relative_transform'].rotation
            
            f.write(f"{relation['marker1_id']},{relation['marker2_id']},"
                   f"{trans[0]},{trans[1]},{trans[2]},"
                   f"{rot[0]},{rot[1]},{rot[2]},{rot[3]},"
                   f"{relation['confidence']},{relation['observation_count']}\n")

def convert_to_reference_frame(relations: Dict[Tuple[int, int], Dict], reference_marker: int) -> str:
    """
    Convert marker relations to a coordinate system centered on a reference marker.
    
    Args:
        relations: Dictionary of marker relations from process_recording
        reference_marker: ID of the marker to use as reference frame
    
    Returns:
        String containing the formatted output
    """
    # Initialize output with header
    output = [
        f"# Reference marker ID: {reference_marker}",
        "# Format: marker_id,tx,ty,tz,qx,qy,qz,qw,confidence,observations"
    ]
    
    # Add reference marker (always at origin with identity rotation)
    output.append(f"{reference_marker},0,0,0,0,0,0,1,1,0")
    
    # Get all unique marker IDs
    marker_ids = set()
    for m1, m2 in relations.keys():
        marker_ids.add(m1)
        marker_ids.add(m2)
    
    # For each marker (except reference), find its transform relative to reference
    for marker_id in marker_ids:
        if marker_id == reference_marker:
            continue
            
        # Try direct relation
        direct_key = (min(reference_marker, marker_id), max(reference_marker, marker_id))
        if direct_key in relations:
            relation = relations[direct_key]
            transform = relation['relative_transform']
            # If reference is second marker in relation, invert transform
            if direct_key[0] == reference_marker:
                transform = Transform.inverse(transform)
            
            # Format output line
            output.append(f"{marker_id},"
                         f"{transform.translation[0]},{transform.translation[1]},{transform.translation[2]},"
                         f"{transform.rotation[0]},{transform.rotation[1]},{transform.rotation[2]},{transform.rotation[3]},"
                         f"{relation['confidence']},{relation['observation_count']}")
    
    return "\n".join(output)

# Example usage:
if __name__ == "__main__":
    # Process a recording file with a specific confidence threshold
    relations = process_recording("recording.csv", min_confidence=0.002)
    
    # Convert to reference frame centered on marker 3
    reference_output = convert_to_reference_frame(relations, reference_marker=3)
    
    # Save to file
    with open("reference_frame.csv", "w") as f:
        f.write(reference_output)
    

    