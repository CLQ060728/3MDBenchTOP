# Video evaluations
# Author: Qian Liu

import os


def run_evalcrafter_metrics(project_root, data_path, metric_name, gpu_id, vid_set_name, output_path_root):
    evalcrafter_path = os.path.join(project_root, "ds_profiling/EvalCrafter/")
    if metric_name == "VQA":
        import builtins
        builtins.EVAL_CRAFTER_PATH_ = evalcrafter_path
        from .EvalCrafter.metrics.DOVER import evaluate_a_set_of_videos as easv
        easv.compute_video_quality_scores(data_path, output_path_root, evalcrafter_path, gpu_id)
    elif metric_name == "IS":
        num_splits = 10
        import builtins
        builtins.EVAL_CRAFTER_PATH_ = evalcrafter_path
        from .EvalCrafter.metrics import inception_score as inceps
        inceps.compute_inception_score(data_path, num_splits, output_path_root, gpu_id)
    elif metric_name == "flow_score" or metric_name == "warping_error":
        from .EvalCrafter.metrics.RAFT import optical_flow_scores as ofs
        ofs.compute_optical_flow_scores(data_path, metric_name, gpu_id,
                                        evalcrafter_path, output_path_root)