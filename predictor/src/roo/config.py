# Constants from paper
HEATMAP_PIXEL_SIZE = 1
EPSILON = 0.3  # Threshold factor for centroid selection
MAX_ITERATIONS = 100  # Maximum K-means iterations
CONVERGENCE_THRESHOLD = 1e-4  # Centroid movement threshold for convergence
OUTLIER_THRESHOLD_MULTIPLIER = 3  # Threshold multiplier for outlier detection

# Adaptive K selection
K_CANDIDATES = [2, 3, 4, 5, 6]  # Candidate values for adaptive K selection (min 2)
USE_ADAPTIVE_K = True  # Enable adaptive K selection (uses Davies-Bouldin Index)
DEFAULT_K = 4  # Default K value when adaptive selection is disabled
