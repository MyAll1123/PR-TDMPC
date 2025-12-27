import numpy as np
from typing import Dict, Tuple, Any

# Kitchen task constants from humanoid_bench

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([2, 3]),
    "top burner": np.array([6, 7]),
    "light switch": np.array([8, 9]),
    "slide cabinet": np.array([10]),
    "hinge cabinet": np.array([11, 12]),
    "microwave": np.array([13]),
    "kettle": np.array([14, 15, 16, 17, 18, 19, 20]),
}

OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1, 0.99, 0.0, 0.0, -0.06]),
}

BONUS_THRESH = 0.3
TASK_ELEMENTS = ["microwave", "kettle", "bottom burner", "light switch"]
ALL_TASKS = [
    "bottom burner",
    "top burner", 
    "light switch",
    "slide cabinet",
    "hinge cabinet",
    "microwave",
    "kettle",
]

def compare_h1hand_kitchen_v0_trajectories(
    trajectory_a: Dict[str, Any],
    trajectory_b: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compare two kitchen task trajectories and return preference.
    
    Args:
        trajectory_a: First trajectory data
        trajectory_b: Second trajectory data  
        config: Configuration parameters (optional)
        
    Returns:
        Tuple of (better_trajectory, worse_trajectory)
    """
    if config is None:
        config = {}
    
    try:
        score_a = _compute_trajectory_score(trajectory_a, config)
        score_b = _compute_trajectory_score(trajectory_b, config)
        
        # Ensure scores are scalar values
        if hasattr(score_a, 'item'):
            score_a = score_a.item()
        if hasattr(score_b, 'item'):
            score_b = score_b.item()
        
        if float(score_a) > float(score_b):
            return trajectory_a, trajectory_b
        else:
            return trajectory_b, trajectory_a
    except Exception as e:
        # Fallback: return trajectories in original order if comparison fails
        return trajectory_a, trajectory_b

def evaluate_dpo_preference(
    trajectory_a: Dict[str, Any],
    trajectory_b: Dict[str, Any], 
    config: Dict[str, Any]
) -> Tuple[str, float]:
    """
    Evaluate DPO preference between two trajectories.
    
    Args:
        trajectory_a: First trajectory data
        trajectory_b: Second trajectory data
        config: Configuration parameters
        
    Returns:
        Tuple of (preferred_trajectory, confidence_score)
    """
    score_a = _compute_trajectory_score(trajectory_a, config)
    score_b = _compute_trajectory_score(trajectory_b, config)
    
    # Calculate confidence based on score difference
    score_diff = abs(score_a - score_b)
    max_possible_diff = 10.0  # Estimated maximum score difference
    confidence = min(score_diff / max_possible_diff, 1.0)
    
    preferred = 'A' if score_a > score_b else 'B'
    return preferred, confidence

def _compute_trajectory_score(trajectory: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Compute comprehensive score for a kitchen trajectory.
    
    Args:
        trajectory: Trajectory data containing observations and rewards
        config: Configuration parameters
        
    Returns:
        Comprehensive trajectory score
    """
    traj_data = _get_trajectory_data(trajectory)
    
    # Handle config parameter safely
    if config is None or not hasattr(config, 'get'):
        config = {}
    
    # Extract configuration weights
    weights = config.get('task_specific_weights', {
        'task_completion': 1.0,
        'task_efficiency': 0.5,
        'task_progress': 0.3,
        'survival_bonus': 0.2
    })
    
    # Calculate individual components
    task_completion_score = _calculate_task_completion(traj_data, config)
    task_efficiency_score = _calculate_task_efficiency(traj_data, config) 
    task_progress_score = _calculate_task_progress(traj_data, config)
    survival_score = _calculate_survival_bonus(traj_data, config)
    
    # Weighted combination
    total_score = (
        weights['task_completion'] * task_completion_score +
        weights['task_efficiency'] * task_efficiency_score +
        weights['task_progress'] * task_progress_score +
        weights['survival_bonus'] * survival_score
    )
    
    return total_score

def _get_trajectory_data(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant data from trajectory.
    
    Args:
        trajectory: Raw trajectory data
        
    Returns:
        Processed trajectory data
    """
    # Handle different trajectory formats
    if hasattr(trajectory, 'get'):
        observations = trajectory.get('observations', [])
        rewards = trajectory.get('rewards', [])
        actions = trajectory.get('actions', [])
    else:
        # Fallback for non-dict trajectory formats
        observations = getattr(trajectory, 'observations', [])
        rewards = getattr(trajectory, 'rewards', [])
        actions = getattr(trajectory, 'actions', [])
    
    if not observations:
        return {
            'obj_positions': [],
            'rewards': [],
            'actions': [],
            'episode_length': 0
        }
    
    # Extract object positions from observations
    obj_positions = []
    for obs in observations:
        if isinstance(obs, (list, np.ndarray)) and len(obs) > 21:
            # Object positions are after robot DOF (21 dimensions)
            obj_pos = obs[21:] if len(obs) > 21 else []
            obj_positions.append(obj_pos)
        else:
            # For shorter observations, create dummy object positions
            obj_positions.append(np.zeros(30))  # Dummy object positions
    
    return {
        'obj_positions': obj_positions,
        'rewards': rewards,
        'actions': actions,
        'episode_length': len(observations)
    }

def _calculate_task_completion(traj_data: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Calculate task completion score based on completed subtasks.
    
    Args:
        traj_data: Processed trajectory data
        config: Configuration parameters
        
    Returns:
        Task completion score (0-1)
    """
    obj_positions = traj_data['obj_positions']
    if not obj_positions:
        return 0.0
    
    # Check final state for task completion
    final_obj_pos = obj_positions[-1]
    if not final_obj_pos or len(final_obj_pos) < 21:
        return 0.0
    
    completed_tasks = 0
    total_tasks = len(TASK_ELEMENTS)
    
    for task in TASK_ELEMENTS:
        if task in OBS_ELEMENT_INDICES and task in OBS_ELEMENT_GOALS:
            element_idx = OBS_ELEMENT_INDICES[task]
            element_goal = OBS_ELEMENT_GOALS[task]
            
            # Check if indices are within bounds
            if all(idx < len(final_obj_pos) for idx in element_idx):
                current_pos = np.array([final_obj_pos[idx] for idx in element_idx])
                distance = np.linalg.norm(current_pos - element_goal)
                
                if distance < BONUS_THRESH:
                    completed_tasks += 1
    
    return completed_tasks / total_tasks if total_tasks > 0 else 0.0

def _calculate_task_efficiency(traj_data: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Calculate task efficiency based on completion time and actions.
    
    Args:
        traj_data: Processed trajectory data
        config: Configuration parameters
        
    Returns:
        Task efficiency score (0-1)
    """
    episode_length = traj_data['episode_length']
    max_episode_steps = config.get('thresholds', {}).get('max_episode_steps', 500)
    
    if episode_length == 0:
        return 0.0
    
    # Efficiency is inversely related to episode length
    # Shorter episodes (faster completion) get higher scores
    efficiency = max(0.0, 1.0 - (episode_length / max_episode_steps))
    
    return efficiency

def _calculate_task_progress(traj_data: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Calculate task progress score based on distance improvements.
    
    Args:
        traj_data: Processed trajectory data
        config: Configuration parameters
        
    Returns:
        Task progress score (0-1)
    """
    obj_positions = traj_data['obj_positions']
    if len(obj_positions) < 2:
        return 0.0
    
    initial_distances = []
    final_distances = []
    
    for task in TASK_ELEMENTS:
        if task in OBS_ELEMENT_INDICES and task in OBS_ELEMENT_GOALS:
            element_idx = OBS_ELEMENT_INDICES[task]
            element_goal = OBS_ELEMENT_GOALS[task]
            
            # Initial distance
            try:
                max_idx = int(np.max(element_idx)) if len(element_idx) > 0 else 0
                if (obj_positions[0] and len(obj_positions[0]) > max_idx and
                    all(idx < len(obj_positions[0]) for idx in element_idx)):
                    initial_pos = np.array([obj_positions[0][idx] for idx in element_idx])
                    initial_dist = np.linalg.norm(initial_pos - element_goal)
                    initial_distances.append(initial_dist)
                else:
                    initial_distances.append(1.0)  # Default high distance
            except (IndexError, ValueError):
                initial_distances.append(1.0)
            
            # Final distance
            try:
                max_idx = int(np.max(element_idx)) if len(element_idx) > 0 else 0
                if (obj_positions[-1] and len(obj_positions[-1]) > max_idx and
                    all(idx < len(obj_positions[-1]) for idx in element_idx)):
                    final_pos = np.array([obj_positions[-1][idx] for idx in element_idx])
                    final_dist = np.linalg.norm(final_pos - element_goal)
                    final_distances.append(final_dist)
                else:
                    final_distances.append(1.0)  # Default high distance
            except (IndexError, ValueError):
                final_distances.append(1.0)
    
    if not initial_distances or not final_distances:
        return 0.0
    
    # Calculate average improvement
    improvements = []
    for init_dist, final_dist in zip(initial_distances, final_distances):
        if init_dist > 0:
            improvement = max(0.0, (init_dist - final_dist) / init_dist)
            improvements.append(improvement)
    
    return np.mean(improvements) if improvements else 0.0

def _calculate_survival_bonus(traj_data: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Calculate survival bonus based on episode length.
    
    Args:
        traj_data: Processed trajectory data
        config: Configuration parameters
        
    Returns:
        Survival bonus score (0-1)
    """
    episode_length = traj_data['episode_length']
    min_survival_time = config.get('thresholds', {}).get('min_survival_time', 50)
    max_episode_steps = config.get('thresholds', {}).get('max_episode_steps', 500)
    
    if episode_length < min_survival_time:
        return 0.0
    
    # Linear bonus for surviving longer
    survival_ratio = min(episode_length / max_episode_steps, 1.0)
    return survival_ratio

# Public interface functions
def compute_kitchen_reward_components(trajectory: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute individual reward components for kitchen task.
    
    Args:
        trajectory: Trajectory data
        config: Configuration parameters
        
    Returns:
        Dictionary of reward components
    """
    traj_data = _get_trajectory_data(trajectory)
    
    return {
        'task_completion': _calculate_task_completion(traj_data, config),
        'task_efficiency': _calculate_task_efficiency(traj_data, config),
        'task_progress': _calculate_task_progress(traj_data, config),
        'survival_bonus': _calculate_survival_bonus(traj_data, config)
    }

def compare_kitchen_trajectories(
    trajectory_a: Dict[str, Any],
    trajectory_b: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two kitchen trajectories with detailed analysis.
    
    Args:
        trajectory_a: First trajectory data
        trajectory_b: Second trajectory data
        config: Configuration parameters
        
    Returns:
        Detailed comparison results
    """
    components_a = compute_kitchen_reward_components(trajectory_a, config)
    components_b = compute_kitchen_reward_components(trajectory_b, config)
    
    score_a = _compute_trajectory_score(trajectory_a, config)
    score_b = _compute_trajectory_score(trajectory_b, config)
    
    return {
        'preferred_trajectory': 'A' if score_a > score_b else 'B',
        'score_a': score_a,
        'score_b': score_b,
        'components_a': components_a,
        'components_b': components_b,
        'score_difference': abs(score_a - score_b)
    }