import numpy as np
import scipy.ndimage
import scipy.optimize
from .centers import find_centers_cpp


def detection_scores(
        truth,
        test,
        label_ids,
        matching_score='overlap',
        matching_threshold=0,
        voxel_size=None,
        return_matches=False):
    '''Compute common detection scores for connected components between two
    arrays. Scores are computed for each label in `label_ids` separately and
    for all labels together.

    Args:

        truth (ndarray):

            Array of true labels.

        test (ndarray):

            Array of predicted labels.

        label_ids (array-like):

            The labels to evaluate. For each ID in this array, connected
            components will be extracted from `truth` and `test` and matched
            with each other.

        matching_score (string, optional):

            Which score to use for the matching of components. Defaults to
            `overlap`, i.e., the number of elements both components have in
            common. Other options are `iou` (intersection over union) and
            `distance` (Euclidean distance between component centers, see also
            `voxel_size`).

        matching_threshold (float, optional):

            If given, only pairs of truth/test components with at least this
            value are considered for matching.

        voxel_size (tuple of int, optional):

            Used to compute Euclidean distances between component centers.
            Affects matching if `distance` is chosen as matching score and
            reported values for average distance between matched components.

        return_matches (bool, optional):

            If set, the returned dictionary will also contain arrays with
            connected components for each label in `truth` and `test`, together
            with a list of matched components between the two.

    Returns:

        Dictionary with the keys:

            `tp`: number of true positives
            `fp`: number of false positives
            `fn`: number of false negatives
            `avg_distance`: average distance between centers of matched
                            components
            `avg_iou`: average intersection-over-union between matched
                       components

        Each of the scores will also be computed per label in `label_ids`,
        resulting in keys `tp_<label_id>`, `fp_<label_id>`, etc.

        If `return_matches` is set, the dictionary will also contain:

            `components_truth_<label_id>`: array of connected components with
                                           label `<label_id>` in `truth`
            `components_test_<label_id>`: array of connected components with
                                          label `<label_id>` in `truth`
            `matches_<label_id>`: a list of tuples matching components from
                                  `test` to `truth`
    '''

    dims = len(truth.shape)
    detection_scores = {
        'tp': 0,
        'fp': 0,
        'fn': 0,
        'avg_distance': 0.0,
        'avg_iou': 0.0
    }

    for label_id in label_ids:

        # get connected components of this label
        test_components, n_test = scipy.ndimage.label(test == label_id)
        true_components, n_true = scipy.ndimage.label(truth == label_id)

        # get sizes
        test_ids, test_counts = np.unique(
            test_components[test_components > 0].ravel(),
            return_counts=True)
        test_sizes = {i: c for i, c in zip(test_ids, test_counts)}
        true_ids, true_counts = np.unique(
            true_components[true_components > 0].ravel(),
            return_counts=True)
        true_sizes = {i: c for i, c in zip(true_ids, true_counts)}

        # get centers
        test_centers = find_centers(test_components, test_ids)
        true_centers = find_centers(true_components, true_ids)
        if voxel_size is not None:
            if n_test > 0:
                test_centers *= voxel_size
            if n_true > 0:
                true_centers *= voxel_size

        # get pairs and count of shared elements (excluding background 0)
        both_fg_mask = np.logical_and(test_components > 0, true_components > 0)
        both_fg_test = test_components[both_fg_mask].ravel()
        both_fg_true = true_components[both_fg_mask].ravel()
        if both_fg_true.size > 0:
            pairs, counts = np.unique(
                np.array([both_fg_test, both_fg_true]),
                axis=1,
                return_counts=True)
        else:
            pairs = np.array([[], []], dtype=test_components.dtype)
            counts = np.array([], dtype=np.int32)

        # get IoUs (for overlapping components, in matrix form)
        ious = np.zeros(
            (n_test + 1, n_true + 1),
            dtype=np.float32)
        ious[pairs[0], pairs[1]] = [
            c/(test_sizes[test_id] + true_sizes[true_id] - c)
            for test_id, true_id, c in zip(pairs[0], pairs[1], counts)
        ]

        # get distances (for all pairs of components, in matrix form)
        distances = np.ones(
            (n_test + 1, n_true + 1),
            dtype=np.float32)
        if n_test > 0 and n_true > 0:
            center_dists = np.sqrt(np.sum(
                np.array([
                    np.subtract.outer(test_centers[:, d], true_centers[:, d])
                    for d in range(dims)
                ])**2,
                axis=0))
            distances *= center_dists.max()*10
            distances[1:, 1:] = center_dists

        # select matching score
        if matching_score == 'overlap':

            # get overlaps (in matrix form)
            overlaps = np.zeros(
                (n_test + 1, n_true + 1),
                dtype=np.int64)
            overlaps[pairs[0], pairs[1]] = counts

            scores = overlaps
            maximize = True

        elif matching_score == 'iou':
            scores = ious
            maximize = True
        elif matching_score == 'distance':
            scores = distances
            maximize = False
        else:
            raise RuntimeError(f"Unknown matching score {matching_score}")

        matches = scipy.optimize.linear_sum_assignment(
            scores,
            maximize=maximize)

        rel = {
            'overlap': np.greater_equal,
            'iou': np.greater_equal,
            'distance': np.less_equal
        }[matching_score]

        # filter matches
        matches = [
            (test_id, true_id)
            for test_id, true_id in zip(matches[0], matches[1])
            if rel(scores[test_id, true_id], matching_threshold)
            and test_id > 0
            and true_id > 0
        ]

        tp = len(matches)
        fp = n_test - tp
        fn = n_true - tp

        if tp > 0:
            avg_distance = np.mean([
                distances[test_id, true_id]
                for test_id, true_id in matches
            ])
            avg_iou = np.mean([
                ious[test_id, true_id]
                for test_id, true_id in matches
            ])
        else:
            avg_distance = 0
            avg_iou = 0

        detection_scores[f'tp_{label_id}'] = tp
        detection_scores[f'fp_{label_id}'] = fp
        detection_scores[f'fn_{label_id}'] = fn
        detection_scores[f'avg_distance_{label_id}'] = avg_distance
        detection_scores[f'avg_iou_{label_id}'] = avg_iou

        detection_scores['tp'] += tp
        detection_scores['fp'] += fp
        detection_scores['fn'] += fn

        detection_scores['avg_distance'] += avg_distance
        detection_scores['avg_iou'] += avg_iou

        if return_matches:

            detection_scores[f'matches_{label_id}'] = matches
            detection_scores[f'components_truth_{label_id}'] = true_components
            detection_scores[f'components_test_{label_id}'] = test_components

    if detection_scores['tp'] > 0:
        detection_scores['avg_distance'] /= detection_scores['tp']
        detection_scores['avg_iou'] /= detection_scores['tp']
    else:
        detection_scores['avg_distance'] = np.nan
        detection_scores['avg_iou'] /= np.nan

    return detection_scores


def find_centers_scipy(components, ids):
    return np.array(scipy.ndimage.measurements.center_of_mass(
            np.ones_like(components),
            components,
            ids))


def find_centers(components, ids):

    if len(components.shape) == 3:

        centers = find_centers_cpp(components.astype(np.uint64))
        return np.array([
            [centers[i]['z'], centers[i]['y'], centers[i]['x']]
            for i in ids
        ])

    else:

        return find_centers_scipy(components, ids)
