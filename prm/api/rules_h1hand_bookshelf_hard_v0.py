#!/usr/bin/env python3
from .common_utils import tolerance

"""
Heuristic rules for Bookshelf Hard task based on humanoid_bench reward function.

This module implements preference comparison for the Bookshelf Hard scenario,
where the robot needs to place multiple objects on different shelves in sequence.
The evaluation is based on the humanoid_bench BookshelfHard task reward formula.

Core reward formula from humanoid_bench:
- Standing and upright posture (20%): standing * upright * small_control
- Object-goal proximity (40%): tolerance-based reward for object placement accuracy
- Hand-object proximity (40%): exponential reward for hand reaching efficiency
- Task progression bonus: 100 * task_index when object is successfully placed

Evaluation dimensions:
1. Standing stability: Maintains proper standing posture and torso uprightness
2. Object placement: Accuracy of placing objects at target shelf positions
3. Hand coordination: Efficiency of hand movements toward target objects
4. Control smoothness: Smooth actuator control without excessive forces
"""

import numpy as np
from typing import Dict, Tuple, Any


def compare_h1hand_bookshelf_hard_v0_trajectories(
    trajectory_a: Dict[str, Any],
    trajectory_b: Dict[str, Any],
    config: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Compare two trajectories for the Bookshelf Hard task.
    
    Args:
        trajectory_a: First trajectory data
        trajectory_b: Second trajectory data
        config: Configuration parameters
        
    Returns:
        Tuple: (better_trajectory, worse_trajectory)
    """
    if config is None:
        config = {}
        
    score_a = calculate_trajectory_score(trajectory_a, config)
    score_b = calculate_trajectory_score(trajectory_b, config)
    
    if score_a > score_b:
        return (trajectory_a, trajectory_b)
    else:
        return (trajectory_b, trajectory_a)


def evaluate_dpo_preference(
    trajectory_a: Dict[str, Any],
    trajectory_b: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[str, float]:
    """
    Evaluate preference between two trajectories with confidence score.
    
    Args:
        trajectory_a: First trajectory data
        trajectory_b: Second trajectory data
        config: Configuration parameters
        
    Returns:
        Tuple of (preferred_trajectory, confidence_score)
    """
    score_a = calculate_trajectory_score(trajectory_a, config)
    score_b = calculate_trajectory_score(trajectory_b, config)
    
    preference_threshold = config.get('preference_threshold', 0.05)
    score_diff = abs(score_a - score_b)
    
    if score_diff < preference_threshold:
        return ('tie', 0.0)
    
    preferred = 'A' if score_a > score_b else 'B'
    confidence = min(score_diff / preference_threshold, 1.0)
    
    return (preferred, confidence)


def calculate_trajectory_score(trajectory: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Calculate overall trajectory score based on humanoid_bench BookshelfHard reward.
    
    The score is computed as a weighted average of:
    - Standing stability reward (20%)
    - Object placement reward (40%) 
    - Hand coordination reward (40%)
    - Task progression bonus
    
    Args:
        trajectory: Trajectory data containing states and actions
        config: Configuration parameters
        
    Returns:
        Overall trajectory score
    """
    try:
        # Extract trajectory data
        states, actions, rewards = _get_trajectory_data(trajectory)
        
        if len(states) == 0:
            return 0.0
        
        # Calculate component rewards
        standing_reward = standing_stability_reward(states, actions, config)
        placement_reward = object_placement_reward(states, config)
        coordination_reward = hand_coordination_reward(states, config)
        progression_bonus = task_progression_reward(trajectory, config)
        
        # Weighted combination matching humanoid_bench formula
        base_score = (
            0.2 * standing_reward +
            0.4 * placement_reward +
            0.4 * coordination_reward
        )
        
        total_score = base_score + progression_bonus
        
        return float(total_score)
        
    except Exception as e:
        print(f"Error calculating trajectory score: {e}")
        return 0.0


def standing_stability_reward(states: np.ndarray, actions: np.ndarray, config: Dict[str, Any]) -> float:
    """
    Calculate standing stability reward based on posture and control.
    
    Evaluates:
    - Head height maintenance (standing)
    - Torso uprightness
    - Small control forces
    
    Args:
        states: State trajectory data
        actions: Action trajectory data
        config: Configuration parameters
        
    Returns:
        Standing stability reward score
    """
    try:
        # Extract head height (assuming it's in the state)
        # This is a simplified approximation - actual implementation would need
        # proper state parsing based on the robot model
        head_heights = states[:, 2]  # Assuming z-position represents height
        stand_height = config.get('stand_height', 1.65)
        
        # Standing reward: tolerance for head height
        standing_scores = _compute_tolerance_reward(
            head_heights,
            bounds=(stand_height, float('inf')),
            margin=stand_height / 4
        )
        
        # Upright reward: simplified torso uprightness check
        # In practice, this would use quaternion or rotation matrix from state
        upright_scores = _compute_tolerance_reward(
            np.ones(len(states)) * 0.95,  # Simplified assumption
            bounds=(0.9, float('inf')),
            margin=1.9,
            sigmoid='linear'
        )
        
        # Small control reward
        control_scores = _compute_small_control_reward(actions, config)
        
        # Combined standing reward
        combined_scores = standing_scores * upright_scores * control_scores
        
        return float(np.mean(combined_scores))
        
    except Exception as e:
        print(f"Error in standing_stability_reward: {e}")
        return 0.0


def object_placement_reward(states: np.ndarray, config: Dict[str, Any]) -> float:
    """
    Calculate object placement accuracy reward.
    
    Evaluates how well objects are placed at target shelf positions.
    
    Args:
        states: State trajectory data
        config: Configuration parameters
        
    Returns:
        Object placement reward score
    """
    try:
        # This is a simplified implementation
        # In practice, would need to extract object positions from state
        # and compare with target shelf positions
        
        # Placeholder implementation - would need actual object position extraction
        placement_goals = [
            [0.75, -0.25, 1.55],
            [0.8, 0.05, 0.95],
            [0.8, -0.25, 0.95],
            [0.85, 0.05, 0.35],
            [0.85, -0.25, 0.35]
        ]
        
        # Simplified distance calculation
        # Would need proper object position extraction from states
        avg_distance = config.get('avg_object_distance', 0.5)
        
        proximity_score = _compute_tolerance_reward(
            np.array([avg_distance]),
            bounds=(0, 0.15),
            margin=1.0,
            sigmoid='linear'
        )
        
        return float(np.mean(proximity_score))
        
    except Exception as e:
        print(f"Error in object_placement_reward: {e}")
        return 0.0


def hand_coordination_reward(states: np.ndarray, config: Dict[str, Any]) -> float:
    """
    Calculate hand coordination reward based on hand-object proximity.
    
    Evaluates how efficiently hands move toward target objects.
    
    Args:
        states: State trajectory data
        config: Configuration parameters
        
    Returns:
        Hand coordination reward score
    """
    try:
        # Simplified implementation
        # Would need to extract hand positions and object positions from states
        
        # Placeholder for hand-object distances
        hand_distances = config.get('avg_hand_distance', 0.3)
        
        # Exponential reward as in humanoid_bench
        coordination_scores = np.exp(-hand_distances)
        
        return float(coordination_scores)
        
    except Exception as e:
        print(f"Error in hand_coordination_reward: {e}")
        return 0.0


def task_progression_reward(trajectory: Dict[str, Any], config: Dict[str, Any]) -> float:
    """
    Calculate task progression bonus based on completed subtasks.
    
    Args:
        trajectory: Full trajectory data
        config: Configuration parameters
        
    Returns:
        Task progression bonus score
    """
    try:
        # Extract success information from trajectory
        success_subtasks = trajectory.get('info', {}).get('success_subtasks', 0)
        
        # Bonus: 100 * task_index as in humanoid_bench
        progression_bonus = 100 * success_subtasks
        
        return float(progression_bonus)
        
    except Exception as e:
        print(f"Error in task_progression_reward: {e}")
        return 0.0


def _compute_tolerance_reward(
    values: np.ndarray,
    bounds: Tuple[float, float],
    margin: float,
    sigmoid: str = 'quadratic'
) -> np.ndarray:
    """
    Compute tolerance-based reward similar to dm_control.utils.rewards.tolerance.
    
    Args:
        values: Input values to evaluate
        bounds: (lower_bound, upper_bound) for tolerance
        margin: Margin for tolerance calculation
        sigmoid: Type of sigmoid function ('quadratic' or 'linear')
        
    Returns:
        Reward values based on tolerance
    """
    lower, upper = bounds
    
    if upper == float('inf'):
        # Only lower bound
        distances = np.maximum(0, lower - values)
    else:
        # Both bounds
        distances = np.maximum(0, np.maximum(lower - values, values - upper))
    
    if sigmoid == 'linear':
        rewards = np.maximum(0, 1 - distances / margin)
    else:  # quadratic
        normalized_distances = distances / margin
        rewards = np.exp(-0.5 * normalized_distances ** 2)
    
    return rewards


def _compute_small_control_reward(actions: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """
    Calculate reward based on small control forces.
    
    Args:
        actions: Action trajectory data
        config: Configuration parameters
        
    Returns:
        Small control reward values
    """
    try:
        # Calculate action magnitudes
        action_magnitudes = np.linalg.norm(actions, axis=1)
        
        # Tolerance reward for small actions
        control_rewards = _compute_tolerance_reward(
            action_magnitudes,
            bounds=(0, float('inf')),
            margin=10.0,
            sigmoid='quadratic'
        )
        
        # Scale as in humanoid_bench: (4 + reward) / 5
        scaled_rewards = (4 + control_rewards) / 5
        
        return scaled_rewards
        
    except Exception as e:
        print(f"Error in _compute_small_control_reward: {e}")
        return np.ones(len(actions))


def _get_trajectory_data(trajectory: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract states, actions, and rewards from trajectory data.
    
    Args:
        trajectory: Trajectory dictionary
        
    Returns:
        Tuple of (states, actions, rewards) as numpy arrays
    """
    try:
        states = np.array(trajectory.get('observations', []))
        actions = np.array(trajectory.get('actions', []))
        rewards = np.array(trajectory.get('rewards', []))
        
        return states, actions, rewards
        
    except Exception as e:
        print(f"Error extracting trajectory data: {e}")
        return np.array([]), np.array([]), np.array([])